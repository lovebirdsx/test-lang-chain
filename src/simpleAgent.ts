import 'dotenv/config';

import { createAgent, tool } from 'langchain';
import fs from 'fs';
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';
import path from 'node:path';

import { NodeSDK } from "@opentelemetry/sdk-node";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { CallbackHandler } from '@langfuse/langchain';
import { propagateAttributes, startActiveObservation } from '@langfuse/tracing';

const sdk = new NodeSDK({
    spanProcessors: [new LangfuseSpanProcessor()],
});

sdk.start();

async function test() {
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
                filePath: z.string().describe('Relative file path'),
                content: z.string().describe('Content'),
            }),
        },
    );

    const systemPrompt = `你是一名专业的游戏开发者，你的工作是根据用户的需求编写游戏代码。`;
    const llm = new ChatOpenAI({
        modelName: 'glm-5',
        temperature: 0,
        streaming: true,
        configuration: {
            baseURL: 'https://open.bigmodel.cn/api/paas/v4',
            apiKey: process.env.GLM_API_KEY,
        }
    });

    const agent = createAgent({
        model: llm,
        tools: [writeFile],
        systemPrompt,
    });

    const langfuseHandler = new CallbackHandler();
    // const consoleHandler = new ConsoleCallbackHandler();

    await startActiveObservation('simple-agent-observation', async () => {
        await propagateAttributes({
            userId: 'shawn',
            sessionId: `simple-agent-${Date.now()}`,
            tags: ['langchain-test', 'simple-agent'],
        }, async () => {
            const result = await agent.invoke(
                { messages: [{ role: 'user', content: '请你生成一个俄罗斯方块游戏' }] },
                {
                    callbacks: [langfuseHandler, {
                        handleLLMNewToken(token: string) {
                            process.stdout.write(token);
                        },
                    }],
                },
            );
            
            console.log('\n\n--- 执行结束 ---');
        });
    });
}

test().catch((error) => {
    console.error('Error running research agent:', error);
});
