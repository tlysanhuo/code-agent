This dataset contains 36083 trajectories. A 25000 subset was used to train SERA-32B. Data was generated from the *second* rollout of SVG on 121 SWE-smith codebases using GLM-4.6 as teacher.

**Schema**:
```
messages: Generated trajectory
instance_id: ID of trajectory
rollout_patch: Created patch to the codebase from the current trajectory
func_name: Name of function sampled from codebase to start the pipeline
func_path: File path to the sampled function
problem_statement: Problem statement provided to the model
target_patch: Ground truth patch (empty if T1) 
docker_image: Docker image used
```

**Verification**:  
Verification can be done on T2 trajectories by comparing generated rollout patches against the target ground truth patch from T1 trajectories.  
We do not verify in our main experiments but provide the metadata to do so in `target_patch` and `rollout_patch`. 

**Note**: Apply json.loads() to the messages column to load.

Sera-4.6-Lite-T2 is licensed under the Open Data Commons Attribution License v1.0 (ODC-By). It is intended for research and educational use. For more information, please see our Responsible Use Guidelines.