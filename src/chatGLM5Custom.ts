import { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel, BaseChatModelCallOptions } from "@langchain/core/language_models/chat_models";
import { ChatResult, ChatGeneration, ChatGenerationChunk } from "@langchain/core/outputs";
import { convertToOpenAITool } from "@langchain/core/utils/function_calling";
import OpenAI from "openai";
import { BaseMessage, AIMessageChunk, AIMessage } from "langchain";

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

    async _generate(
        messages: BaseMessage[],
        _options: this["ParsedCallOptions"],
        _runManager?: CallbackManagerForLLMRun
    ): Promise<ChatResult> {
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
            throw new Error("GLM 返回为空，无法生成结果。");
        }

        const aiMessage = buildAIMessageFromResponse(choice);
        const generation: ChatGeneration = {
            text: typeof aiMessage.content === "string" ? aiMessage.content : stringifyContent(aiMessage.content),
            message: aiMessage,
        };

        return {
            generations: [generation],
            llmOutput: {
                model: this.modelName,
                finish_reason: choice.finish_reason,
            },
        };
    }

    async *_streamResponseChunks(
        messages: BaseMessage[],
        _options: this["ParsedCallOptions"],
        runManager?: CallbackManagerForLLMRun
    ): AsyncGenerator<ChatGenerationChunk> {
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

        for await (const chunk of stream) {
            const choice = chunk.choices?.[0];
            const delta: any = choice?.delta;
            const finishReason = choice?.finish_reason;

            if (!delta) continue;

            const reasoningText =
                (delta.reasoning_content ?? delta.reasoning ?? "") as string;

            if (reasoningText) {
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

                    // 让 LangChain callback 体系也感知到“新片段到了”
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
    }
}