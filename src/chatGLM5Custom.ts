import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel, BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";
import { ChatResult, ChatGeneration, ChatGenerationChunk } from "@langchain/core/outputs";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import OpenAI from "openai";
import { BaseMessage, AIMessageChunk, AIMessage } from "langchain";

import { startObservation } from "@langfuse/tracing";

type ToolLike = any;

type GLMChatModelInput = {
    apiKey: string;
    model: string;
    temperature?: number;
    baseURL?: string;
    streamUsage?: boolean;
};

type PartialToolState = {
    id?: string;
    name?: string;
    argsText: string;
    index: number;
};

export function stringifyContent(content: unknown): string {
    if (typeof content === "string") return content;
    if (content == null) return "";
    try {
        return JSON.stringify(content);
    } catch {
        return String(content);
    }
}

function safeJsonParse<T = unknown>(text: string, fallback: T): T {
    try {
        return JSON.parse(text) as T;
    } catch {
        return fallback;
    }
}

function getMessageType(message: BaseMessage): string {
    const anyMsg = message as any;
    if (typeof anyMsg.getType === "function") return anyMsg.getType();
    if (typeof anyMsg._getType === "function") return anyMsg._getType();
    return anyMsg.type ?? "human";
}

function buildAIMessageFromResponse(choice: any): AIMessage {
    const message = choice?.message ?? {};
    const rawToolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];

    const tool_calls = rawToolCalls.map((tc: any) => ({
        id: tc.id,
        type: "tool_call" as const,
        name: tc.function?.name ?? "",
        args: safeJsonParse<Record<string, unknown>>(tc.function?.arguments ?? "{}", {}),
    }));

    return new AIMessage({
        content: message.content ?? "",
        tool_calls,
        additional_kwargs: {
            reasoning_content:
                message.reasoning_content ??
                message.reasoning ??
                undefined,
        },
        response_metadata: {
            finish_reason: choice?.finish_reason,
        },
    });
}

function toOpenAIMessages(messages: BaseMessage[]) {
    return messages.map((message) => {
        const type = getMessageType(message);
        const anyMsg = message as any;

        if (type === "system") {
            return {
                role: "system" as const,
                content: stringifyContent(message.content),
            };
        }

        if (type === "human") {
            return {
                role: "user" as const,
                content: stringifyContent(message.content),
            };
        }

        if (type === "tool") {
            return {
                role: "tool" as const,
                content: stringifyContent(message.content),
                tool_call_id: anyMsg.tool_call_id ?? anyMsg.toolCallId,
            };
        }

        if (type === "ai") {
            const toolCalls = anyMsg.tool_calls ?? anyMsg.toolCalls;
            if (Array.isArray(toolCalls) && toolCalls.length > 0) {
                return {
                    role: "assistant" as const,
                    content: stringifyContent(message.content),
                    tool_calls: toolCalls.map((tc: any) => ({
                        id: tc.id,
                        type: "function",
                        function: {
                            name: tc.name,
                            arguments: JSON.stringify(tc.args ?? {}),
                        },
                    })),
                };
            }

            return {
                role: "assistant" as const,
                content: stringifyContent(message.content),
            };
        }

        return {
            role: "user" as const,
            content: stringifyContent(message.content),
        };
    });
}

export class ChatGLM5Custom extends BaseChatModel<BaseChatModelCallOptions> {
    private readonly client: OpenAI;
    private readonly modelName: string;
    private readonly temperature: number;
    private readonly streamUsage: boolean;
    private readonly baseURL: string;
    private readonly apiKey: string;
    private readonly boundTools: ToolLike[];

    constructor(fields: GLMChatModelInput & { boundTools?: ToolLike[] }) {
        super({});
        this.apiKey = fields.apiKey;
        this.baseURL = fields.baseURL ?? "https://open.bigmodel.cn/api/paas/v4";
        this.modelName = fields.model;
        this.temperature = fields.temperature ?? 0;
        this.streamUsage = fields.streamUsage ?? false;
        this.boundTools = fields.boundTools ?? [];

        this.client = new OpenAI({
            apiKey: this.apiKey,
            baseURL: this.baseURL,
        });
    }

    _llmType(): string {
        return "glm-5-custom-langchain";
    }

    bindTools(tools: ToolLike[]) {
        return new ChatGLM5Custom({
            apiKey: this.apiKey,
            baseURL: this.baseURL,
            model: this.modelName,
            temperature: this.temperature,
            streamUsage: this.streamUsage,
            boundTools: tools,
        });
    }

    private getInvocationTools() {
        if (!this.boundTools.length) return undefined;
        return this.boundTools.map((tool) => convertToOpenAITool(tool));
    }

    /**
     * 将 LangChain messages 和 tools 定义打包为 Langfuse generation 的 input
     * 这就是在 Langfuse 后台能看到的 "Input" 区域
     */
    private buildLangfuseInput(messages: BaseMessage[]) {
        const openAIMessages = toOpenAIMessages(messages);
        const tools = this.getInvocationTools();

        return {
            messages: openAIMessages,
            ...(tools ? { tools } : {}),
            ...(this.boundTools.length ? { tool_choice: "auto" } : {}),
        };
    }

    /**
     * 返回模型调用参数，显示在 Langfuse 的 "Model Parameters" 区域
     */
    private buildModelParameters() {
        return {
            temperature: String(this.temperature),
            stream: "false",
            thinking: "enabled",
        };
    }

    async _generate(
        messages: BaseMessage[],
        _options: this["ParsedCallOptions"],
        _runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {

        // asType: "generation" 让 Langfuse 知道这是一次 LLM 调用
        const generation = startObservation(
            "glm-5-generation",
            {
                model: this.modelName,
                input: this.buildLangfuseInput(messages),
                modelParameters: this.buildModelParameters(),
            },
            { asType: "generation" }        // 标记为 generation 类型
        );

        try {
            const response = await this.client.chat.completions.create({
                model: this.modelName,
                temperature: this.temperature,
                stream: false,
                messages: toOpenAIMessages(messages) as any,
                tools: this.getInvocationTools() as any,
                tool_choice: this.boundTools.length ? "auto" : undefined,
                thinking: {
                    type: "enabled",
                    clear_thinking: false,
                } as any,
                tool_stream: true as any,
            } as any);

            const choice = response.choices?.[0];
            if (!choice) {
                generation.update({
                    level: "ERROR",
                    statusMessage: "GLM 返回为空",
                }).end();
                throw new Error("GLM 返回为空，无法生成结果。");
            }

            const aiMessage = buildAIMessageFromResponse(choice);
            const generationResult: ChatGeneration = {
                text: typeof aiMessage.content === "string" ? aiMessage.content : stringifyContent(aiMessage.content),
                message: aiMessage,
            };

            // 将输出和 token 用量写入 Langfuse generation
            const usage = (response as any).usage;
            generation.update({
                output: {
                    content: aiMessage.content,
                    tool_calls: aiMessage.tool_calls,
                    reasoning_content: aiMessage.additional_kwargs?.reasoning_content,
                    finish_reason: choice.finish_reason,
                },
                ...(usage ? {
                    usageDetails: {
                        input: usage.prompt_tokens ?? 0,
                        output: usage.completion_tokens ?? 0,
                        total: usage.total_tokens ?? 0,
                    },
                } : {}),
                metadata: {
                    finish_reason: choice.finish_reason ?? "unknown",
                    provider: "zhipu",
                },
            }).end();

            return {
                generations: [generationResult],
                llmOutput: {
                    model: this.modelName,
                    finish_reason: choice.finish_reason,
                },
            };
        } catch (error) {
            // 发生错误时也要正确结束 observation
            generation.update({
                level: "ERROR",
                statusMessage: error instanceof Error ? error.message : String(error),
            }).end();
            throw error;
        }
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        _options: this["ParsedCallOptions"],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {

        const generation = startObservation(
            "glm-5-generation-stream",
            {
                model: this.modelName,
                input: this.buildLangfuseInput(messages),
                modelParameters: {
                    ...this.buildModelParameters(),
                    stream: "true",
                },
            },
            { asType: "generation" }
        );

        const stream = await this.client.chat.completions.create({
            model: this.modelName,
            temperature: this.temperature,
            stream: true,
            messages: toOpenAIMessages(messages) as any,
            tools: this.getInvocationTools() as any,
            tool_choice: this.boundTools.length ? "auto" : undefined,
            thinking: {
                type: "enabled",
                clear_thinking: false,
            } as any,
            tool_stream: true as any,
            stream_options: this.streamUsage ? { include_usage: true } : undefined,
        } as any) as unknown as AsyncIterable<any>;

        const toolState = new Map<number, PartialToolState>();
        let emittedFinalToolCalls = false;

        // 用于收集完整输出，最终写入 generation.output
        let fullContentText = "";
        let fullReasoningText = "";
        let lastFinishReason: string | undefined;
        let completionStarted = false;

        try {
            for await (const chunk of stream) {
                const choice = chunk.choices?.[0];
                const delta: any = choice?.delta;
                const finishReason = choice?.finish_reason;

                if (finishReason) {
                    lastFinishReason = finishReason;
                }

                if (!delta) continue;

                const reasoningText =
                    (delta.reasoning_content ?? delta.reasoning ?? "") as string;

                if (reasoningText) {
                    fullReasoningText += reasoningText;
                    await runManager?.handleLLMNewToken(reasoningText);

                    yield new ChatGenerationChunk({
                        text: "",
                        message: new AIMessageChunk({
                            content: "",
                            additional_kwargs: {
                                reasoning_content: reasoningText,
                            },
                        }),
                    });
                }

                if (delta.content) {
                    const text = delta.content as string;
                    if (!completionStarted) {
                        completionStarted = true;
                        generation.update({
                            completionStartTime: new Date(),
                        });
                    }

                    fullContentText += text;
                    await runManager?.handleLLMNewToken(text);

                    yield new ChatGenerationChunk({
                        text,
                        message: new AIMessageChunk({
                            content: text,
                        }),
                    });
                }

                if (Array.isArray(delta.tool_calls)) {
                    for (const tc of delta.tool_calls) {
                        const index = tc.index ?? 0;
                        const prev: PartialToolState = toolState.get(index) ?? {
                            argsText: "",
                            index,
                        };

                        const id = tc.id ?? prev.id;
                        const name = tc.function?.name ?? prev.name;
                        const argsDelta = tc.function?.arguments ?? "";
                        const argsText = prev.argsText + argsDelta;

                        const nextState: PartialToolState = {
                            id,
                            name,
                            argsText,
                            index,
                        };
                        toolState.set(index, nextState);

                        if (argsDelta) {
                            await runManager?.handleLLMNewToken(argsDelta);
                        }

                        yield new ChatGenerationChunk({
                            text: "",
                            message: new AIMessageChunk({
                                content: "",
                                tool_call_chunks: [
                                    {
                                        type: "tool_call_chunk",
                                        id,
                                        index,
                                        name,
                                        args: argsDelta,
                                    } as any,
                                ],
                            }),
                        });
                    }
                }

                if (finishReason === "tool_calls" && toolState.size > 0 && !emittedFinalToolCalls) {
                    emittedFinalToolCalls = true;

                    const finalToolCalls = [...toolState.values()]
                        .sort((a, b) => a.index - b.index)
                        .map((tc) => ({
                            id: tc.id ?? `call_${tc.index}`,
                            type: "tool_call" as const,
                            name: tc.name ?? "",
                            args: safeJsonParse<Record<string, unknown>>(tc.argsText || "{}", {}),
                        }));

                    yield new ChatGenerationChunk({
                        text: "",
                        message: new AIMessageChunk({
                            content: "",
                            tool_calls: finalToolCalls,
                        }),
                    });
                }

                if (chunk.usage) {
                    generation.update({
                        usageDetails: {
                            input: chunk.usage.prompt_tokens ?? 0,
                            output: chunk.usage.completion_tokens ?? 0,
                            total: chunk.usage.total_tokens ?? 0,
                        },
                    });
                }
            }

            if (toolState.size > 0 && !emittedFinalToolCalls) {
                const finalToolCalls = [...toolState.values()]
                    .sort((a, b) => a.index - b.index)
                    .map((tc) => ({
                        id: tc.id ?? `call_${tc.index}`,
                        type: "tool_call" as const,
                        name: tc.name ?? "",
                        args: safeJsonParse<Record<string, unknown>>(tc.argsText || "{}", {}),
                    }));

                yield new ChatGenerationChunk({
                    text: "",
                    message: new AIMessageChunk({
                        content: "",
                        tool_calls: finalToolCalls,
                    }),
                });
            }

            const collectedToolCalls = toolState.size > 0
                ? [...toolState.values()]
                    .sort((a, b) => a.index - b.index)
                    .map((tc) => ({
                        id: tc.id ?? `call_${tc.index}`,
                        name: tc.name ?? "",
                        arguments: tc.argsText,
                    }))
                : undefined;

            generation.update({
                output: {
                    content: fullContentText || undefined,
                    tool_calls: collectedToolCalls,
                    reasoning_content: fullReasoningText || undefined,
                    finish_reason: lastFinishReason,
                },
                metadata: {
                    finish_reason: lastFinishReason ?? "unknown",
                    provider: "zhipu",
                },
            }).end();

        } catch (error) {
            generation.update({
                level: "ERROR",
                statusMessage: error instanceof Error ? error.message : String(error),
            }).end();
            throw error;
        }
    }
}