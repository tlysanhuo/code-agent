# 数据清洗示例:被拒轨迹(真实样本)

> 扫描池 69,705 条 → 保留 28,208。逐规则淘汰量:超长 37,763 / 非单源 3,711 / 格式 18 / 轮数 2 / 重复 3。

## 示例一:超长(主力规则,占淘汰 91%)

- `oauthlib__oauthlib.1fd52536.combine_file__a0mopj7p` · **18,955 tokens > 16,384** → 拒
- 不截断保留的原因:截断会制造无终止轨迹,教会模型不收尾

首两轮:

**[system]** You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reason…

**[user]** <pr_description>
Consider the following PR description:
# OAuth1 Client class has multiple critical issues

I've discovered several issues with the OAuth1 Client class that cause authentication to fail. The problems appear to be related to parameter handling, signature generation, and request preparation.

## Issue Description

When trying to use the OAuth1 Client for authentication, requests fail…

**[assistant]** THOUGHT: I need to start by understanding the codebase structure and finding the OAuth1 Client class that's mentioned in the PR description. Let me first explore the directory structure to see what files are available.

```bash
find . -type f -name "*.py" | grep -E "(oauth|client)" | head -20
```

## 示例二:非单一来源

- source = `openhands_swe_gym` ≠ klear66k_swe_smith → 跳过(单源单格式原则)

- `python__mypy-10308` · 19,116 tokens

## 示例三:格式违规(仅 18 条,质检作用)

- `tweepy__tweepy.91a41c6e.func_basic__c6thyjo6` · 违规 assistant 回复:THOUGHT 前缀=缺 · bash 代码块=0 个(要求恰好 1)

该回复开头:

```
Perfect! I have successfully identified and fixed the issue described in the PR. Here's a summary of what I accomplished:

## Issue Analysis
The PR described an issue where the `get_owned_lists` method in the `Client` class was using an incorrect HTTP method (`POST` instead of `GET`) when retrieving owned lists for a user.

## Solution Implemented
I fixed the HTTP method in both implementations:

1. **Synchronous Client** (`./tweepy/client.py` line 3269): Changed `"POST"` to `"GET"`
2. **Asynchr
```

## 保留标准(对照)

长度 ≤16k · Klear 单源 · 每个 assistant 回复 = THOUGHT + 恰好一个 bash 块 · 轮数 3-120 · 末轮为正常终止的 assistant · 全文 sha256 去重

完整保留示例见 [example-sft-trajectory.md](example-sft-trajectory.md)。