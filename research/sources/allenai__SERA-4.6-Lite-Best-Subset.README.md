---
license: apache-2.0
---
This dataset contains **47498** high-quality samples from Sera-4.6-Lite-T1 and Sera-4.6-Lite-T2. Training leads to a **open-source SoTA 50.67% +/- 1.86%** performance on SWE-Bench Verified at 32K context length, **outperforming Devstral-Small-2 and GLM-4.5-Air**.

**Method**:  
We keep only model-submitted train samples and then filter by truncation ratio at 32K tokens until a threshold ratio of 0.88.

**Schema**:
```
messages: Generated trajectory
instance_id: ID of trajectory
rollout_patch: Created patch to the codebase from the current trajectory
func_name: Name of function sampled from codebase to start the pipeline
func_path: File path to the sampled function
source: Sera-4.6-Lite-T1 | Sera-4.6-Lite-T2
problem_statement: Problem statement provided to the model
target_patch: Ground truth patch (empty if T1) 
docker_image: Docker image used
```

**Verification**:  
Verification can be done on T2 trajectories by comparing generated rollout patches against the target ground truth patch from T1 trajectories.  
We do not verify in our main experiments but provide the metadata to do so in `target_patch` and `rollout_patch`. 

**Note**: Apply json.loads() to the messages column to load.

Sera-4.6-Lite-47000 is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our Responsible Use Guidelines.