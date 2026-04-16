# Rveda Hackathon Backlog and Validation Notes

Canonical mental model and implementation map: [Rveda Environment](<Rveda Environment.md>)

Context: Rveda was built for the Meta PyTorch OpenEnv Hackathon x SST as a real-world OpenEnv environment for agentic medical coding.

Current public implementation:

- GitHub: https://github.com/anirudw/rveda
- Hugging Face Space: https://huggingface.co/spaces/anirudw/rveda
- Deployed host from Space metadata: https://anirudw-rveda.hf.space

## Resolved From Scratch Notes

- Port is `8000` in `openenv.yaml`, `Dockerfile`, and the Hugging Face Space metadata.
- The Space is Docker-based and uses base path `/web`.
- The current action contract is `SEARCH(query)`, `DETAILS(code)`, and `SUBMIT(code)`.
- The current task IDs are `easy_endo_1`, `medium_endo_1`, and `hard_cardio_1`.
- The OpenEnv client/WebSocket path is the intended multi-step interaction path; raw HTTP reset/step state should be rechecked before relying on it.

## Post-Hackathon Priority Backlog

- [ ] Re-run readiness against the deployed Hugging Face Space after each deployment.
- [ ] Verify `reset -> DETAILS -> SUBMIT` works through the exact grader path.
- [ ] Expand the ICD-10 corpus beyond the current mock records.
- [ ] Upgrade retrieval from SQLite `LIKE` to FTS5 or synonym expansion.
- [ ] Add tests for exact code, same-family code, unrelated code, Excludes1 conflict, timeout, and `/step` response shape.
- [ ] Keep `reward`, `done`, `observation`, and `info` aligned with the grader expectations.
- [ ] Keep rewards normalized to the expected 0-1 range in inference outputs.

## Raw Scratch Notes

[ ] check if the ports are supposed to be 8000 
[ ] One important caveat: this changes the runtime contract from returning a bare MedicalObservation to returning a dict 
  payload. That matches your grader requirement, but you should redeploy and re-run the readiness script against HF to 
  confirm the OpenEnv server wrapper serializes it the way you expect.  




`CODEX`

 Plan                                                                                                                        
  1. Fix session semantics first.                                                                                      
     The biggest product risk is that plain HTTP /reset and /step do not share state under the OpenEnv wrapper. Decide 
     which interaction mode the hackathon grader uses.                                                                 
     If it is WebSocket/OpenEnv client based, document that clearly and test only that path for multi-step episodes.   
     If it is plain HTTP, implement explicit session persistence for HTTP or redesign the environment so each /step is 
     self-contained.                                                                                                   
     Success criterion: reset -> DETAILS -> SUBMIT works reliably in one episode.                                      
  2. Convert “difficulty labels” into real graders.                                                                    
     Right now you have 3 tasks, but only 1 shared reward rule.                                                        
     Implement per-task grader logic:
                                                                                                                       
  - Easy: exact code + simple family partial credit                                                                    
  - Medium: exact code + synonym/search evidence expectations                                                          
  - Hard: exact code + exclusion/conflict checks + evidence trail                                                      
    Keep all outward rewards in [0.0, 1.0].                                                                            
    Success criterion: each task has distinct evaluation logic and deterministic expected scores.                      
                                                                                                                       
  3. Separate penalty signals from normalized reward.                                                                  
     Keep reward in [0.0, 1.0], but move richer judge signals into info or observation metadata:                       
                                                                                                                       
  - excludes1_penalty                                                                                                  
  - insufficient_evidence                                                                                              
  - wrong_family                                                                                                       
  - used_search                                                                                                        
    This preserves compliance while still giving useful grading signals.                                               
                                                                                                                       
  4. Strengthen the retrieval layer.                                                                                   
     The current DB/corpus is too small for convincing medical behavior.                                               
     Expand data/icd10.db or regenerate it from a much richer source.                                                  
     At minimum, add coverage for:                                                                                     
                                                                                                                       
  - hypertension                                                                                                       
  - blood pressure                                                                                                     
  - epistaxis                                                                                                          
  - obesity variants                                                                                                   
  - thyroid disorders
  - MI/chest pain differentials                                                                                        
    If hackathon reviewers expect “real-world task,” the corpus needs to feel materially larger.                       
                                                                                                                       
  5. Improve search quality.                                                                                           
     Current search is simple LIKE.                                                                                    
     Upgrade to either:                                                                                                
                                                                                                                       
  - SQLite FTS5 with indexed descriptions and aliases                                                                  
  - or a lightweight synonym expansion layer before SQL                                                                
    This is the most visible product-quality improvement after state handling.                                         
                                                                                                                       
  6. Make task reset deterministic and testable.                                                                       
     Expose task selection cleanly and verify it works over the actual deployed interface.                             
     Use task IDs consistently:                                                                                        
                                                                                                                       
  - easy_endo_1                                                                                                        
  - medium_endo_1                                                                                                      
  - hard_cardio_1                                                                                                      
    Success criterion: each task can be targeted and scored predictably in tests and demos.                            
                                                                                                                       
  7. Align API contract with grader expectations.                                                                      
     Keep /step returning exactly                                                                                                               
  - observation                                                                                                        
  - reward                                                                                                             
  - done                                                                                                               
  - info                                                                                                               
    Do not leak extra top-level keys.                                                                                  
    Make sure info is always present, even if empty                                                                                                               
  1. Build a proper automated test suite.                                                                              
     Add pytest coverage for:                                                                                                                                                                                                           
  - task reset by ID                                                                                                   
  - exact reward 1.0                                                                                                   
  - partial reward 0.5                                                                                                 
  - unrelated code reward 0.0                                                                                          
  - Excludes1 flag in metadata/info                                                                                    
  - reward always within [0.0, 1.0]                                                                                    
  - /step response shape exactness                                                                                     
  - search latency threshold                                                                                           
    Also add one end-to-end client-session test using the OpenEnv client path.
  1. Harden deployment validation.
     Before every push:                                                                                                                                                                                                
  - openenv validate                                                                                                   
  - local smoke test                                                                                                   
  - Docker build                                                                                                       
  - readiness audit against deployed Space                                                                             
    Recommended sequence:                                                                                                                                                                                                    
  
  ``` 
  openenv validate                                                   
  docker build -t rveda-env:latest .                                                                                   
  python check-readiness.py  
  ```                                                                                          
           
  1. Improve presentation for hackathon judging.                                                                      
+ Update README to make the project feel intentional:       
  - explain the 3 tasks and why they represent increasing difficulty                                                   
  - show action space and observation space clearly                                                                    
  - document normalized reward semantics                                                                               
  - document why info carries grader diagnostics                                                                       
  - include a short demo flow for one task 
---

  ## Execution Order                                                       
  1. Session/state fix                                                                                                 
  2. Real task-specific graders                                                                                        
  3. Reward/info cleanup                                                                                               
  4. Retrieval corpus expansion                                                                                        
  5. Search upgrade                                                                                                    
  6. Tests                                                                                                             
  7. README/demo polish
  8. openenv validate, build, deploy, verify
---

  ## Recommendation             
  If time is limited, do these three before anything else:
  9. verify session correctness on the actual grader path
  10. make the 3 tasks truly distinct in grading behavior   
  11. expand the database so search looks credible                         
  If you want, I can turn this into a concrete implementation backlog with file-by-file changes and acceptance criteria.    
---
## Tags

#rveda 