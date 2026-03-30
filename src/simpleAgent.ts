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

async function langfuseRun(test: () => Promise<void>) {
    const sdk = new NodeSDK({
        spanProcessors: [new LangfuseSpanProcessor()],
    });

    sdk.start();

    try {
        await startActiveObservation('simple-agent-observation', async () => {
            await propagateAttributes({
                userId: 'shawn',
                sessionId: `simple-agent-${new Date().toLocaleString('zh-CN').replace(/[-\/\s:]/g, '')}`,
                tags: ['langchain-test', 'simple-agent'],
            }, test)
        });
    } catch (err) {
        console.error('测试执行错误:', err);
    } finally {
        sdk.shutdown();
    }
}

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
        },
    }).bindTools([writeFile]);

    const agent = createAgent({
        model: llm,
        tools: [writeFile],
        systemPrompt,
    });

    const langfuseHandler = new CallbackHandler();
    const eventStream = agent.streamEvents({
        messages: [{ role: 'user', content: '请你生成一个俄罗斯方块游戏' }],
    }, {
        version: 'v2',
        callbacks: [langfuseHandler],
    });

    for await (const event of eventStream) {
        const eventType = event.event;

        if (eventType === 'on_chat_model_stream') {
            const chunk = event.data.chunk;
            if (chunk?.content) {
                process.stdout.write(chunk.content);
            }
        } else if (eventType === "on_tool_start") {
            console.log(`\n\n[🤔 思考阶段: 决定使用工具 '${event.name}']`);
            console.log(`[📦 传入参数]: ${JSON.stringify(event.data.input).substring(0, 200)}...\n`);
        } else if (eventType === "on_tool_end") {
            console.log(`\n[✅ 工具执行完毕] 结果: ${JSON.stringify(event.data.output)}\n`);
        } else if (eventType === "on_tool_error") {
            console.error(`\n[❌ 工具执行失败]: ${event.data.error}\n`);
        }
    }

    console.log('\n\n--- 执行结束 ---');
}

langfuseRun(test);
