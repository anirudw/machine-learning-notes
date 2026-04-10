# Rveda Hackathon Quick Runtime Notes

Canonical project documentation: [Rveda Environment](<Rveda Environment.md>)

Rveda was built for the Meta PyTorch OpenEnv Hackathon x SST. These are the short runtime notes for rebuilding and checking the environment.

The short version for Rveda is:

```powershell
docker build -t rveda-env:latest -f Dockerfile .
docker run -p 8000:8000 rveda-env:latest
```

The public Space also uses port `8000` with Docker and base path `/web`.

## Raw Notes

OLMo 3 Base - Full LLM Lifecycle
<u>openenv</u>
	Use agents
Docker file config : https://www.youtube.com/live/kkCNMz0Ptd8?si=b7qgj6PbVpSMYQ_7&t=4357
Use codex : 52 mins

## Add Dependencies
Python dependencies in pyproject.toml
Others in Dockerfile

Add the line to docker file for proper UI display in hugging face





## RUNNING

`docker build -t rveda:latest .`

`docker run -p 8000:8000 rveda:latest`


## REQUIREMENTS
OpenAI test
