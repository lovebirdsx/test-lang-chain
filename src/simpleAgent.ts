import 'dotenv/config';

import { createAgent, tool } from 'langchain';
import fs from 'fs';
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';
import path from 'node:path';
import { ProxyAgent, setGlobalDispatcher } from 'undici';

async function test() {
    const proxyUrl = 'http://localhost:7897';
    const dispatcher = new ProxyAgent(proxyUrl);
    setGlobalDispatcher(dispatcher);

    const workingDir = path.resolve('workspace/simple-agent');
    const writeFile = tool(
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
            await fs.promises.writeFile(fullPath, content, 'utf-8');
            return { success: true, path: fullPath };
        },
        {
            name: 'write_file',
            description: 'Write content to a file',
            schema: z.object({
                filePath: z.string().describe('The file path to write to'),
                content: z.string().describe('The content to write to the file'),
            }),
        },
    );
    
    const systemPrompt = `你是一名专业的游戏开发者，你的工作是根据用户的需求编写游戏代码。

## \`write_file\`
使用此工具将内容写入文件。你可以指定文件路径和要写入的内容。
`;

    const llm = new ChatOpenAI({
        modelName: 'gemini-3.1-pro-preview', 
        temperature: 0,
        configuration: {
            baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/', 
            apiKey: process.env.GEMINI_API_KEY,
        }
    });

    const agent = createAgent({
        model: llm,
        tools: [writeFile],
        systemPrompt,
    });

    const stream = await agent.streamEvents(
        { messages: [{ role: 'user', content: '请你生成一个俄罗斯方块游戏' }] },
        { version: 'v2' } // 必须指定 version: 'v2'
    );

    for await (const event of stream) {
        const eventType = event.event;
        
        if (eventType === 'on_chat_model_stream') {
            const content = event.data.chunk?.content;
            if (content && typeof content === 'string') {
                process.stdout.write(content);
            }
        } else if (eventType === 'on_tool_start') {
            console.log(`\n\n[🔧 开始调用工具]: ${event.name}`);
            // console.log('[📥 工具输入]:', JSON.stringify(event.data.input, null, 2));
        } else if (eventType === 'on_tool_end') {
            console.log(`\n[✅ 工具调用结束]: ${event.name}`);
            console.log('[📤 工具输出]:', event.data.output);
        } else {
            // console.log(`\n[ℹ️ 事件]: ${eventType}`);
        }

    }

    console.log('\n\n--- 执行结束 ---');
}

test().catch((error) => {
    console.error('Error running research agent:', error);
});
