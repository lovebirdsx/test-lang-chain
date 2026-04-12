import 'dotenv/config';

import { tool } from 'langchain';
import { TavilySearch } from '@langchain/tavily';
import { z } from 'zod';
import { createDeepAgent } from 'deepagents';
import { ChatOpenAI } from '@langchain/openai';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { propagateAttributes, startActiveObservation } from '@langfuse/tracing';
import { CallbackHandler } from '@langfuse/langchain';

async function langfuseRun(test: () => Promise<void>) {
    const sdk = new NodeSDK({
        spanProcessors: [new LangfuseSpanProcessor()],
    });

    sdk.start();

    try {
        await startActiveObservation("simple-deep-agent-observation", async () => {
            await propagateAttributes(
                {
                    userId: "shawn",
                    sessionId: `simple-deep-agent-${new Date()
                        .toLocaleString("zh-CN")
                        .replace(/[-/\\s:]/g, "")}`,
                    tags: ["langchain-test", "simple-deep-agent"],
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


async function test() {
    const internetSearch = tool(
        async ({
            query,
            maxResults = 5,
            topic = 'general',
            includeRawContent = false,
        }: {
            query: string;
            maxResults?: number;
            topic?: 'general' | 'news' | 'finance';
            includeRawContent?: boolean;
        }) => {
            const tavilySearch = new TavilySearch({
                maxResults,
                tavilyApiKey: process.env.TAVILY_API_KEY,
                includeRawContent,
                topic,
            });
            return await tavilySearch._call({ query });
        },
        {
            name: 'internet_search',
            description: 'Run a web search',
            schema: z.object({
                query: z.string().describe('The search query'),
                maxResults: z
                    .number()
                    .optional()
                    .default(5)
                    .describe('Maximum number of results to return'),
                topic: z
                    .enum(['general', 'news', 'finance'])
                    .optional()
                    .default('general')
                    .describe('Search topic category'),
                includeRawContent: z
                    .boolean()
                    .optional()
                    .default(false)
                    .describe('Whether to include raw content'),
            }),
        },
    );
    
    const llm = new ChatOpenAI({
        modelName: process.env.OPENAI_MODEL, 
        temperature: 0,
        configuration: {
            baseURL: process.env.OPENAI_BASE_URL, 
            apiKey: process.env.OPENAI_API_KEY 
        }
    });
    
    const systemPrompt = '你是一个搜索问答助手，有必要时，你会通过搜索网络来获取最新的信息来回答用户的问题';
    const langfuseHandler = new CallbackHandler();
    const agent = createDeepAgent({
        model: llm,
        tools: [internetSearch],
        systemPrompt,
    });

    const stream = await agent.streamEvents(
        { messages: [{ role: 'user', content: '未来一周广州的天气' }] },
        { version: 'v2', callbacks: [langfuseHandler] } // 必须指定 version: 'v2'
    );

    for await (const event of stream) {
        const eventType = event.event;

        if (eventType === 'on_chat_model_stream') {
            const content = event.data.chunk?.content;
            if (content && typeof content === 'string') {
                process.stdout.write(content);
            }
        }
        else if (eventType === 'on_tool_start') {
            console.log(`\n\n[🔧 开始调用工具]: ${event.name}`);
            console.log('[📥 工具输入]:', JSON.stringify(event.data.input, null, 2));
        }
        else if (eventType === 'on_tool_end') {
            console.log(`\n[✅ 工具调用结束]: ${event.name}`);
            console.log('[📤 工具输出]:', event.data.output);
        }
    }

    console.log('\n\n--- 执行结束 ---');
}

langfuseRun(test);
