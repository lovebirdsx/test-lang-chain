import "dotenv/config";

import { createAgent, tool } from "langchain";
import fs from "fs";
import path from "node:path";
import { z } from "zod";

import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { CallbackHandler } from "@langfuse/langchain";
import { propagateAttributes, startActiveObservation } from "@langfuse/tracing";

import { ChatGLM5Custom, stringifyContent } from "./chatGLM5Custom";
import { formatDateTime } from "./common";

const FIRST_EVENT_NOTICE_DELAY_MS = 800;
const TOOL_ARGS_FLUSH_INTERVAL_MS = 150;
const TOOL_RESULT_PREVIEW_LIMIT = 1200;

function safeStringify(value: unknown, maxLen = TOOL_RESULT_PREVIEW_LIMIT): string {
    try {
        const text =
            typeof value === "string" ? value : JSON.stringify(value, null, 2);
        return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
    } catch {
        const text = String(value);
        return text.length > maxLen ? `${text.slice(0, maxLen)}...` : text;
    }
}

const ARG_INLINE_LIMIT = 200;
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/**
 * 将工具参数格式化为字段级摘要，长值只显示类型+长度+预览
 */
function summarizeArgs(args: unknown): string {
    if (args == null || typeof args !== "object") {
        return `  ${safeStringify(args, ARG_INLINE_LIMIT)}`;
    }

    const entries = Object.entries(args as Record<string, unknown>);
    if (entries.length === 0) return "  (无参数)";

    return entries
        .map(([key, value]) => {
            const strValue = typeof value === "string" ? value : JSON.stringify(value ?? "");
            if (strValue.length <= ARG_INLINE_LIMIT) {
                return `  ${key}: ${strValue}`;
            }
            const typeName = typeof value === "string" ? "string" : typeof value;
            const preview = strValue.slice(0, 80).replace(/\n/g, "\\n");
            return `  ${key}: [${typeName}, ${strValue.length.toLocaleString()} chars] ${preview}...`;
        })
        .join("\n");
}

async function langfuseRun(test: () => Promise<void>) {
    const sdk = new NodeSDK({
        spanProcessors: [new LangfuseSpanProcessor()],
    });

    sdk.start();

    try {
        await startActiveObservation("simple-agent-observation", async () => {
            await propagateAttributes(
                {
                    userId: "shawn",
                    sessionId: `simple-agent-${formatDateTime()}`,
                    tags: ["langchain-test", "simple-agent"],
                },
                test
            );
        });
    } catch (err) {
        console.error("测试执行错误:", err);
    } finally {
        sdk.shutdown();
    }
}

function createWriteFileTool(workingDir: string) {
    return tool(
        async ({
            filePath,
            content,
        }: {
            filePath: string;
            content: string;
        }) => {
            const fullPath = path.resolve(workingDir, filePath);
            const dir = path.dirname(fullPath);
            await fs.promises.mkdir(dir, { recursive: true });
            await fs.promises.writeFile(fullPath, content, "utf-8");
            return { success: true, path: fullPath };
        },
        {
            name: "write_file",
            description: "Write content to a file",
            schema: z.object({
                filePath: z.string().describe("Relative file path"),
                content: z.string().describe("Content"),
            }),
        }
    );
}

async function test() {
    const workingDir = path.resolve("workspace/simple-agent");
    const writeFile = createWriteFileTool(workingDir);
    const systemPrompt =
        "你是一名专业的游戏开发者，你的工作是根据用户的需求编写游戏代码。";

    const llm = new ChatGLM5Custom({
        model: "glm-5",
        temperature: 0,
        apiKey: process.env.GLM_API_KEY!,
        baseURL: "https://open.bigmodel.cn/api/paas/v4",
    }).bindTools([writeFile]);

    const agent = createAgent({
        model: llm,
        tools: [writeFile],
        systemPrompt,
    });

    const langfuseHandler = new CallbackHandler();

    let hasObservedActivity = false;
    let toolBannerShown = false;

    let currentToolIndex: number | null = null;
    let currentToolName = "";
    let lastFlushAt = 0;
    let spinnerTick = 0;

    let contentAccum = "";           // 累积所有 content 文本
    let contentIsToolJson = false;   // 是否已进入工具 JSON 区域
    let contentToolJsonLen = 0;      // 工具 JSON 部分的累计长度

    const toolArgBuffers = new Map<number, string>();
    const toolArgPrintedLengths = new Map<number, number>();

    process.stdout.write("\n[请求已提交，正在等待模型响应...]\n");

    const firstEventFallbackTimer = setTimeout(() => {
        if (!hasObservedActivity) {
            process.stdout.write("[模型正在分析需求，可能即将调用工具...]\n");
        }
    }, FIRST_EVENT_NOTICE_DELAY_MS);

    function flushToolArgs(index: number, force = false) {
        const fullText = toolArgBuffers.get(index) ?? "";
        const printedLen = toolArgPrintedLengths.get(index) ?? 0;
        const pending = fullText.slice(printedLen);

        if (!pending) return;

        const now = Date.now();
        if (!force && now - lastFlushAt < TOOL_ARGS_FLUSH_INTERVAL_MS) {
            return;
        }

        const frame = SPINNER_FRAMES[spinnerTick % SPINNER_FRAMES.length];
        spinnerTick++;

        // 只显示单行进度，用 \r 覆盖刷新，不输出参数原文
        process.stdout.write(
            `\r${frame} [工具参数生成中] ${currentToolName || `tool#${index}`} | ${fullText.length.toLocaleString()} chars`
        );

        if (force) {
            process.stdout.write("\n");
        }

        toolArgPrintedLengths.set(index, fullText.length);
        lastFlushAt = now;
    }

    try {
        const stream = await agent.stream(
            {
                messages: [{ role: "user", content: "请你生成一个俄罗斯方块游戏" }],
            },
            {
                streamMode: ["messages", "updates"],

                callbacks: [langfuseHandler],
            }
        );

        for await (const [mode, chunk] of stream as AsyncIterable<[string, any]>) {
            if (!hasObservedActivity) {
                hasObservedActivity = true;
                clearTimeout(firstEventFallbackTimer);
            }

            if (mode === "messages") {
                const [messageChunk, metadata] = chunk;

                const reasoning =
                    messageChunk?.additional_kwargs?.reasoning_content ??
                    messageChunk?.kwargs?.additional_kwargs?.reasoning_content ??
                    "";

                if (reasoning) {
                    process.stdout.write(reasoning);
                }

                const contentText =
                    typeof messageChunk?.content === "string"
                        ? messageChunk.content
                        : messageChunk?.content
                            ? stringifyContent(messageChunk.content)
                            : messageChunk?.text ?? "";

                if (contentText) {
                    contentAccum += contentText;

                    if (!contentIsToolJson) {
                        // 检测 content 流是否进入了工具调用 JSON 区域
                        // GLM 系列模型会在正常文本后，直接在 content 中输出工具调用的 JSON
                        const searchStart = Math.max(0, contentAccum.length - contentText.length - 50);
                        const searchSlice = contentAccum.slice(searchStart);
                        const jsonStartPattern = /\{"(?:filePath|file_path|path|name|code|content|query|input)"\s*:/;
                        const match = searchSlice.match(jsonStartPattern);

                        if (match && match.index !== undefined) {
                            // 找到了工具 JSON 起始位置
                            const absoluteIndex = searchStart + match.index;
                            contentIsToolJson = true;
                            contentToolJsonLen = contentAccum.length - absoluteIndex;

                            // JSON 起始之前的部分仍然正常输出
                            const jsonStartInChunk = contentText.length - (contentAccum.length - absoluteIndex);
                            if (jsonStartInChunk > 0) {
                                process.stdout.write(contentText.slice(0, jsonStartInChunk));
                            }

                            process.stdout.write("\n[正在生成文件内容...]\n");
                        } else {
                            // 还没检测到 JSON，正常输出
                            process.stdout.write(contentText);
                        }
                    } else {
                        // 已在工具 JSON 区域，抑制原文输出，只显示进度
                        contentToolJsonLen += contentText.length;

                        const frame = SPINNER_FRAMES[spinnerTick % SPINNER_FRAMES.length];
                        spinnerTick++;
                        process.stdout.write(
                            `\r${frame} [文件内容生成中] ${contentToolJsonLen.toLocaleString()} chars`
                        );
                    }
                }

                const toolCallChunks =
                    messageChunk?.tool_call_chunks ??
                    messageChunk?.kwargs?.tool_call_chunks ??
                    [];

                if (Array.isArray(toolCallChunks) && toolCallChunks.length > 0) {
                    if (!toolBannerShown) {
                        toolBannerShown = true;
                        process.stdout.write("\n[模型正在生成工具调用参数...]\n");
                    }

                    for (const tc of toolCallChunks) {
                        const index = tc?.index ?? 0;
                        const name = tc?.name ?? currentToolName ?? "";
                        const argsDelta = tc?.args ?? "";

                        if (currentToolIndex !== index) {
                            currentToolIndex = index;
                            currentToolName = name || `tool#${index}`;
                            process.stdout.write(`\n[候选工具] ${currentToolName}\n`);
                        } else if (name && name !== currentToolName) {
                            currentToolName = name;
                        }

                        if (argsDelta) {
                            const prev = toolArgBuffers.get(index) ?? "";
                            toolArgBuffers.set(index, prev + argsDelta);
                            flushToolArgs(index, false);
                        }
                    }
                }
            }

            if (mode === "updates") {
                const updates = chunk;

                for (const [step, data] of Object.entries(updates ?? {})) {
                    const lastMessage = (data as any)?.messages?.at?.(-1);

                    if (
                        lastMessage?.tool_calls &&
                        Array.isArray(lastMessage.tool_calls) &&
                        lastMessage.tool_calls.length > 0
                    ) {
                        for (const tc of lastMessage.tool_calls) {
                            const toolName = tc.name ?? "unknown_tool";
                            const argsSummary = summarizeArgs(tc.args);

                            process.stdout.write(
                                `\n[工具调用已定稿] ${toolName}\n${argsSummary}\n`
                            );
                        }
                    }

                    if (lastMessage?._getType?.() === "tool" || lastMessage?.tool_call_id) {
                        process.stdout.write(
                            `\n[工具步骤完成: ${step}] ${safeStringify(lastMessage.content)}\n`
                        );

                        contentIsToolJson = false;
                        contentAccum = "";
                        contentToolJsonLen = 0;
                    }
                }
            }
        }

        if (currentToolIndex != null) {
            flushToolArgs(currentToolIndex, true);
        }
    } finally {
        clearTimeout(firstEventFallbackTimer);
    }

    console.log("\n\n--- 执行结束 ---");
}

langfuseRun(test);