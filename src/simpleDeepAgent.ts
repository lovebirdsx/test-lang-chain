import 'dotenv/config';

import { tool } from 'langchain';
import { TavilySearch } from '@langchain/tavily';
import { z } from 'zod';
import { createDeepAgent } from 'deepagents';
import { ChatOpenAI } from '@langchain/openai';
import { propagateAttributes } from '@langfuse/core';
import { LangfuseSpanProcessor } from '@langfuse/otel';
import { startActiveObservation } from '@langfuse/tracing';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { formatDateTime } from './common';
import { CallbackHandler } from '@langfuse/langchain';

async function langfuseRun(test: () => Promise<void>) {
    const sdk = new NodeSDK({
        spanProcessors: [new LangfuseSpanProcessor({
            shouldExportSpan: () => true
        })],
    });

    sdk.start();

    try {
        await startActiveObservation("simple-agent-observation", async () => {
            await propagateAttributes(
                {
                    userId: "shawn",
                    sessionId: `simple-deep-agent-${formatDateTime()}`,
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

    const researchInstructions = `你是一个查询助手，你可以使用互联网搜索工具来帮助你回答用户的问题。`;
    // const researchInstructions = `你是一名专家研究员。你的工作是进行深入研究，然后撰写一份精美的报告。你可以访问互联网搜索工具作为收集信息的主要手段。`;

    const llm = new ChatOpenAI({
        modelName: 'stepfun/step-3.5-flash:free',
        temperature: 0,
        configuration: {
            baseURL: 'https://openrouter.ai/api/v1',
            apiKey: process.env.OPENROUTER_API_KEY
        }
    });

    const langfuseHandler = new CallbackHandler();
    const agent = createDeepAgent({
        model: llm,
        tools: [internetSearch],
        systemPrompt: researchInstructions,
    });

    const stream = await agent.streamEvents(
        // { messages: [{ role: 'user', content: '请你介绍deepagent?' }] },
        { messages: [{ role: 'user', content: '未来7天广州的天气如何？' }] },
        {
            version: 'v2',
            callbacks: [langfuseHandler]
        }
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

    // const result = await agent.invoke({
    //     messages: [{ role: 'user', content: '请你介绍langgraph?' }],
    // });

    // // Print the agent's response
    // console.log(result.messages[result.messages.length - 1].content);
}

langfuseRun(test);
