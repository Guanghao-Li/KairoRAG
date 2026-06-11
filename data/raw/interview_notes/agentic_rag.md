# 面试笔记：Agentic RAG

Agentic RAG 和 top-k RAG 的区别在于，Agent 可以自主选择检索工具、检查中间 snippet、只在必要时读取完整 chunk，并在证据充分时停止。最终回答应当引用已经读取的 chunk，而不是直接引用原始搜索 snippet。
