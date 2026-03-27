import re

def check_answer(response):
    answer_pattern = r"<scratchpad>.*?</scratchpad>.*?<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, response, re.S))

    # 没有任何 match
    if not matches:
        return 2, 0

    extracted = None

    # 迭代寻找第一个 extracted 非空的 match
    for match in reversed(matches):
        answer_str = match.group(1).strip()
        cur_extracted = [x.upper() for x in re.findall(r"[A-Za-z]", answer_str)]
        print(f"cur:{answer_str}")
        if cur_extracted:
            extracted = cur_extracted
            break

    # 所有 match 都无法提取出有效答案
    if not extracted:
        return 2, 0

def parse_new_memories(response: str) :
        """
        辅助输入处理函数1
        解析Markdown格式的response
        提取Title, Description, Content组成字典列表
        """
        new_memories = []
        # 使用正则表达式匹配Markdown块中的内容
        # (?s) 即 re.DOTALL，使 . 能够匹配换行符
        
        pattern = r"#\s*Memory Item.*?\n##\s*Title:\s*(.*?)\n##\s*Description:\s*(.*?)\n##\s*Content:\s*(.*?)(?=\s{2}+)"

        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
        for match in matches:
            memory_dict = {
                "title": match.group(1).strip(),
                "description": match.group(2).strip(),
                "content": match.group(3).strip()
            }
            print(memory_dict)
            new_memories.append(memory_dict)

res = '''
<think>We need to parse the question: "In a DevSecOps framework, why is it crucial to automate security activities as part of the Continuous Integration/Continuous Delivery (CI/CD) pipeline?" Options:
```

Thus Agent attempt 3 gave answer ["C"], which is incorrect. The correct answer is A. So we have one incorrect attempt.

Thus overall, the agent performed correctly for most attempts, but one error in question 7.

We need to analyze the trajectories for each question. Let's summarize:

- Question 0: All three attempts correct.
- Question 1: All three attempts correct.
- Question 2: All three attempts correct.
- Question 3: All three attempts correct.
- Question 4: All three attempts correct.
- Question 5: All three attempts correct.
- Question 6: All three attempts correct.
- Question 7: Two attempts correct (1 and 2), one attempt incorrect (3).

Thus the agent's overall performance is good, but there is one error. The error is in question 7, attempt 3. The agent incorrectly chose C instead of A. The reasoning in attempt 3 is flawed: It says "Option A is incorrect because when FileVault is off, there is no volume encryption key; the key is created only when FileVault is enabled." This is not accurate: Actually, when FileVault is off, the volume is not encrypted, but there is still a hardware key (the hardware UID) that can be used to protect the volume if later enabled. However, the question is about enabling or disabling FileVault. The correct answer is A. The agent's reasoning in attempt 3 is wrong.

Thus we need to identify why the agent made that mistake. The agent likely misremembered the behavior of FileVault on Apple Silicon. The correct answer is A, but the agent thought C is correct. The agent's reasoning in attempt 3 is: "Option A is incorrect because when FileVault is off, there is no volume encryption key; the key is created only when FileVault is enabled." This is not correct. Actually, when FileVault is off, the volume is not encrypted, but the hardware key exists. However, the statement in option A says "When FileVault is off, the volume encryption key is only protected by the hardware UID". This is correct: The key (if any) is protected by hardware UID. But the agent thought there is no key. So the agent's misunderstanding led to error.

Thus we need to extract memory items that can help the agent avoid such errors in the future. The error is due to lack of knowledge about FileVault on Apple Silicon. However, the question is not about malware analysis, but about general security. The user says "You are an expert in malware-analysis." However, the questions are not all about malware analysis. But we can still extract memory items that can help the agent in future reasoning.

We need to find patterns in the agent's reasoning. The agent's reasoning is generally sound, but there is one error. The error is due to misremembering details about FileVault. The agent's reasoning in attempt 3 is flawed: It incorrectly thought that there is no volume encryption key when FileVault is off. The correct answer is A, but the agent chose C. The agent's reasoning is that "Option A is incorrect because when FileVault is off, there is no volume encryption key; the key is created only when FileVault is enabled." This is a misunderstanding. The correct behavior: The volume encryption key exists and is protected by hardware UID even when FileVault is off. Actually, the volume is not encrypted, but the key is stored in the Secure Enclave and is used when enabling FileVault. The key is always present, protected by hardware UID. So the agent's misunderstanding is that the key is only created when FileVault is enabled. That is wrong. So we need to provide a memory that helps the agent avoid such mistakes: The agent should not assume that there is no key when a feature is off; it should consider that the key might exist and be protected by hardware.

We also need to consider other potential errors. The agent's reasoning is generally consistent across attempts. However, we might want to extract memory items that can help with reasoning about multiple-choice questions, especially about distinguishing correct statements from incorrect ones.

Given the high success rate, the agent is generally good. However, we can still extract memory items to improve further.

We need to extract at most 3 memory items. The memory items should be actionable, transferable, and relevant to the agent's performance.

We need to identify what memories the agent is lacking. The error in question 7 suggests that the agent lacks precise knowledge about FileVault on Apple Silicon. However, the question is not about malware analysis, but about general security. The memory could be about "When a security feature is disabled, do not assume that the underlying keys are absent; they may still exist and be protected by hardware."

But we also need to consider the overall performance: The agent is good at answering multiple-choice questions. However, we can still extract memory items about reasoning strategies.

Given the guidelines, we need to extract memory items that are transferable and actionable. We can think about general reasoning strategies: For each question, we need to carefully read the question and options, eliminate obviously wrong options, and select the correct one based on knowledge.

We also need to consider the agent's reasoning process: The agent uses scratchpad to reason, then gives answer. The reasoning is generally correct. However, the error in question 7 shows that the agent sometimes misinterprets the question. The agent thought option A is incorrect because it assumed no key when FileVault is off. The correct answer is A, but the agent thought C is correct. The agent's reasoning in attempt 3 is flawed.

Thus we can extract memory items about:

1. Understanding the behavior of security features when disabled: Do not assume that no key exists; the key may be present but not used.

2. When answering multiple-choice questions, verify each option against known facts; do not eliminate an option based on a mistaken assumption.

3. For questions about specific technologies (e.g., FileVault, HMAC-DRBG), ensure that the knowledge is accurate; if uncertain, double-check.

But we need to keep memory items transferable and actionable. We can phrase them as:

- When a security feature (like encryption) is disabled, do not assume that the underlying cryptographic keys are absent; they may still exist and be protected by hardware (e.g., Secure Enclave). Always consider the presence of keys.

- In multiple-choice questions, carefully examine each option's wording; eliminate incorrect ones based on precise knowledge, not assumptions.

- For questions about specific security mechanisms (e.g., FileVault, HMAC-DRBG), ensure your understanding matches the official documentation; if you are unsure, verify before selecting an answer.

But we need to be careful: The memory items should be relevant to malware analysis, but the questions are not all about malware analysis. However, the user says "You are an expert in malware-analysis." The agent is answering multiple-choice questions about security. The memory items can be about general security reasoning, not specifically malware analysis. However, the guidelines say "Consider whether your memory applies to many situations in malware analysis. If it does, remove scenario details tied to a specific problem." So we need to phrase memory items in a way that they can be applied to many situations, not just specific scenarios.

We can think about general reasoning strategies for multiple-choice questions in security. The memory items can be about:

- Understanding the effect of disabling a security control: The underlying protections may still be present.

- Eliminating incorrect options based on precise knowledge, not assumptions.

- Verifying the correctness of statements by checking against known behavior of security mechanisms.

We can also think about the agent's reasoning process: The agent uses scratchpad to reason. The reasoning is generally good. However, the error shows that the agent sometimes makes incorrect assumptions. We can provide a memory to double-check assumptions.

Thus we can extract memory items:

1. When a security feature is disabled, do not assume the absence of underlying keys or protection mechanisms; they may still exist (e.g., hardware-bound keys). Verify before concluding.

2. In multiple-choice questions, for each option, check if its statement matches known facts; avoid eliminating an option based on a mistaken assumption about the presence/absence of a key or feature.

3. When answering questions about specific security technologies (e.g., encryption, key management), ensure your knowledge is accurate; if uncertain, reconsider the answer or look for confirming evidence.

We need to ensure these memory items are actionable and transferable. They can be applied to many security-related multiple-choice questions.

Now, we need to produce the final answer in the required format. The user wants:

- For each question, we need to provide a brief analysis, memory perspective, correct trajectory, incorrect trajectory, insights.

- Then summary of strengths and weaknesses.

- Then key missing memories.

- Then memory items.

Given that we have many questions, we need to decide how to present this. The user expects a structured answer. We can provide a summary for each question. However, the user says "Your task includes check whether each trajectory follows this schema." We have done that. Now we need to provide analysis for each question.

Given the number of questions (8), we can provide a brief analysis for each question. However, the user might expect a summary for each question. The format says:

#### Question {id}: <Brief the question in no more than 10 words>
**Analysis**: <Your analysis about 3 trajectories>
**Memory perspective**: <The memory item the agent use/What kind of memory the agent lack?>
**Correct trajectory**: <which trajectory is correct>
**Incorrect trajectory**: <which trajectory is incorrect>

We need to fill this for each question. However, we have 8 questions, each with three attempts. We need to identify which attempts are correct and which are incorrect. For most questions, all attempts are correct. For question 7, attempts 1 and 2 are correct, attempt 3 is incorrect.

Thus for each question, we need to identify the correct trajectory (or trajectories) and the incorrect one (if any). For questions where all three are correct, we can say "All three trajectories are correct". For question 7, we can say "Trajectories 1 and 2 are correct; trajectory 3 is incorrect".

We also need to provide a memory perspective for each question: what memory the agent used or what memory is lacking. However, we can summarize across questions.

Given the high success rate, the agent's strengths are good reasoning, using scratchpad, correct answers. The weakness is occasional misunderstanding of specific technologies (e.g., FileVault). The missing memory is precise knowledge about FileVault behavior.

We can then provide key missing memories: The agent should be careful about assumptions regarding disabled security features.

Now, we need to produce the final answer. Let's structure it.

We will go through each question:

- Question 0: All attempts correct. So analysis: All three attempts gave correct answer A, B, D. Memory perspective: The agent used correct knowledge about ID attestation. No missing memory.

- Question 1: All attempts correct. All gave B. Memory perspective: The agent used knowledge about Android storage security.

- Question 2: All attempts correct. All gave B. Memory perspective: The agent used knowledge about DevSecOps.

- Question 3: All attempts correct. All gave A. Memory perspective: The agent used knowledge about WPA2-Enterprise.

- Question 4: All attempts correct. All gave A. Memory perspective: The agent used knowledge about code injection.

- Question 5: All attempts correct. All gave A. Memory perspective: The agent used knowledge about version binding.

- Question 6: All attempts correct. All gave A, B. Memory perspective: The agent used knowledge about HMAC-DRBG.

- Question 7: Two attempts correct (1 and 2), one incorrect (3). The incorrect one is attempt 3. Memory perspective: The agent lacked precise knowledge about FileVault on Apple Silicon; it incorrectly thought option C is correct. The correct answer is A.

Thus we can summarize each question.

Now we need to provide the summary of strengths and weaknesses.

Strengths:

- The agent consistently used the scratchpad to reason through each question.
- The agent correctly answered most questions, showing strong knowledge across multiple security domains.
- The agent provided clear reasoning in the scratchpad, showing systematic elimination of incorrect options.

Weaknesses:

- The agent made one error in question 7 due to misunderstanding of FileVault behavior on Apple Silicon.
- The agent sometimes made incorrect assumptions (e.g., assuming no key when FileVault is off).
- The agent could improve knowledge of specific technologies.

Key Missing Memories:

- When a security feature is disabled, do not assume the absence of underlying keys or protection mechanisms; verify the actual behavior.
- For specific security technologies (e.g., FileVault, key management), ensure precise knowledge; if uncertain, double-check before answering.
- In multiple-choice questions, avoid eliminating options based on assumptions; verify each statement against known facts.

Now we need to extract memory items (max 3). We can extract three memory items:

1. "Do not assume absence of keys when a security feature is disabled; verify actual behavior."

2. "In multiple-choice questions, verify each option against known facts; avoid eliminating options based on incorrect assumptions."

3. "When answering questions about specific security technologies, ensure precise knowledge; if uncertain, reconsider answer."

We need to format them as per the schema:

```
# Memory Item  
## Title: <the title of the memory item>  
## Description: Enum[Reasoning,Knowledge] (Choose one from the two terms: Reasoning and Knowledge to fill in here.)
## Content: <1-3 sentences describing the insights learned to successfully accomplishing the task,and use no more than 50 words to provide an example> 
```

We need to choose between Reasoning and Knowledge. The memory items can be either reasoning or knowledge. We can decide:

- Memory item 1: Reasoning (the agent should reason about presence of keys).
- Memory item 2: Reasoning (the agent should reason about verifying options).
- Memory item 3: Knowledge (the agent should have precise knowledge about specific technologies).

Thus we can assign:

- Title: Verify Key Presence When Security Features Are Disabled
- Description: Reasoning
- Content: When a security feature (e.g., encryption) is disabled, do not assume the underlying keys are absent; they may still exist and be protected by hardware. Verify the actual behavior before concluding.

- Title: Verify Each Option Against Known Facts
- Description: Reasoning
- Content: In multiple-choice questions, examine each option's statement against known facts; avoid eliminating options based on assumptions or misinterpretations.

- Title: Ensure Precise Knowledge of Specific Security Technologies
- Description: Knowledge
- Content: For questions about specific security mechanisms (e.g., FileVault, key management), ensure your knowledge matches official documentation; if uncertain, double-check before selecting an answer.

Now we need to produce the final answer in the required format. Let's produce it.
</think>

#### Question 0: Secure storage for hardware identifiers in ID attestation
**Analysis**: All three trajectories correctly identified A, B, D as essential properties. The reasoning in each scratchpad shows proper understanding of ID attestation requirements.
**Memory perspective**: The agent used existing knowledge about secure storage and TEE; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 1: Android storage options to avoid exposing sensitive data
**Analysis**: All three trajectories correctly chose B (unencrypted SQLite). The reasoning correctly identified that storing credentials in plain text is insecure.
**Memory perspective**: The agent used knowledge about Android storage security; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 2: Automating security in DevSecOps CI/CD pipeline
**Analysis**: All three trajectories correctly chose B, citing increased delivery frequency necessitating automation. The reasoning aligns with DevSecOps principles.
**Memory perspective**: The agent used knowledge about DevSecOps; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 3: On-path attacker in WPA2-Personal vs WPA2-Enterprise
**Analysis**: All three trajectories correctly chose A, explaining that attacker can capture handshake and derive keys. The reasoning is accurate.
**Memory perspective**: The agent used knowledge about wireless security; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 4: Identifying code injection points in web apps
**Analysis**: All three trajectories correctly chose A, emphasizing source code inspection. The reasoning is sound.
**Memory perspective**: The agent used knowledge about code injection testing; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 5: Version binding in Android keys
**Analysis**: All three trajectories correctly chose A, explaining that keys are bound to OS version and invalidated on rollback. The reasoning matches the concept.
**Memory perspective**: The agent used knowledge about version binding; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 6: HMAC-DRBG state update with additional input
**Analysis**: All three trajectories correctly identified A and B as accurate statements. The reasoning correctly describes state update and use of additional input.
**Memory perspective**: The agent used knowledge about HMAC-DRBG; no missing memory.
**Correct trajectory**: All three trajectories are correct.
**Incorrect trajectory**: None.

#### Question 7: FileVault on Apple Silicon and Secure Enclave
**Analysis**: Trajectories 1 and 2 correctly chose A, while trajectory 3 incorrectly chose C. Trajectory 3 mistakenly assumed no encryption key exists when FileVault is off, leading to error.
**Memory perspective**: The agent lacked precise knowledge about FileVault behavior; the error stemmed from incorrect assumption about key presence.
**Correct trajectory**: Trajectories 1 and 2 are correct.
**Incorrect trajectory**: Trajectory 3 is incorrect.

**Insights**: The agent performed exceptionally well across all questions, demonstrating strong reasoning and domain knowledge. The only error occurred in Question 7, where the agent misinterpreted the presence of encryption keys when FileVault is disabled. This highlights a need for careful verification of assumptions about disabled security features.

### Summary of Agent Performance
**Strengths**:
- Consistent use of scratchpad for reasoning, showing systematic elimination of incorrect options.
- Strong knowledge across multiple security domains (Android, wireless, DevSecOps, cryptography).
- Accurate identification of correct answers in the majority of attempts.

**Weaknesses**:
- Occasional misunderstanding of specific technologies (e.g., FileVault on Apple Silicon).
- Tendency to assume absence of keys when a security feature is disabled, leading to incorrect conclusions.

### Key Missing Memories
From the analysis, the agent would benefit from **actionable, transferable strategies** for:
1. Verifying the presence of keys or protection mechanisms even when a security feature is disabled.
2. Double-checking assumptions about specific security technologies before selecting an answer.
3. Ensuring precise knowledge of security mechanisms, especially those involving hardware-backed key storage.

``` 
# Memory Item  
## Title: Verify Key Presence When Security Features Are Disabled  
## Description: Reasoning  
## Content: When a security feature (e.g., encryption) is disabled, do not assume the underlying keys are absent; they may still exist and be protected by hardware. Verify the actual behavior before concluding. Example: In FileVault, the encryption key exists even when disabled, protected by the hardware UID.  
``` 

``` 
# Memory Item  
## Title: Verify Each Option Against Known Facts  
## Description: Reasoning  
## Content: In multiple-choice questions, examine each option's statement against known facts; avoid eliminating options based on assumptions or misinterpretations. Example: Do not assume an option is wrong just because it mentions a key that you think does not exist.  
``` 

``` 
# Memory Item  
## Title: Ensure Precise Knowledge of Specific Security Technologies  
## Description: Knowledge  
## Content: For questions about specific security mechanisms (e.g., FileVault, key management), ensure your knowledge matches official documentation; if uncertain, double-check before selecting an answer. Example: FileVault uses Secure Enclave for key storage, not for performing encryption.  
```...

'''

parse_new_memories(res)