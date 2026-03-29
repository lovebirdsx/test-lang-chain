import 'dotenv/config';

import { tool } from 'langchain';
import { TavilySearch } from '@langchain/tavily';
import { z } from 'zod';
import { createDeepAgent } from 'deepagents';
import { ChatOpenAI } from '@langchain/openai';

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

    // System prompt to steer the agent to be an expert researcher
    const researchInstructions = `你是一名专家研究员。你的工作是进行深入研究，然后撰写一份精美的报告。
你可以访问互联网搜索工具作为收集信息的主要手段。

## \`internet_search\`
使用此工具运行互联网搜索。你可以指定返回结果的最大数量、搜索主题和是否应包含原始内容。
`;

    const llm = new ChatOpenAI({
        modelName: 'stepfun/step-3.5-flash:free', 
        temperature: 0,
        configuration: {
            baseURL: 'https://openrouter.ai/api/v1', 
            apiKey: process.env.OPENROUTER_API_KEY 
        }
    });

    const agent = createDeepAgent({
        model: llm,
        tools: [internetSearch],
        systemPrompt: researchInstructions,
    });

    const stream = await agent.streamEvents(
        { messages: [{ role: 'user', content: '请你介绍langgraph?' }] },
        { version: 'v2' } // 必须指定 version: 'v2'
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

test().catch((error) => {
    console.error('Error running research agent:', error);
});
