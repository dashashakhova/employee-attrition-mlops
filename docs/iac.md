# Infrastructure as Code

## Local infrastructure

Docker Compose is used as the declarative definition of the local ML platform. It describes PostgreSQL, MinIO, MLflow, Airflow and the FastAPI service.

Start:

```bash
docker compose up --build -d
```

Health and status:

```bash
docker compose ps
docker ps
```

Destroy the infrastructure and persistent volumes after the project:

```bash
docker compose down -v
```

## Cloud infrastructure

`render.yaml` is the declarative Render Blueprint for the public FastAPI service. It defines the Docker build context and `/health` HTTP health check.

Render can apply a Blueprint from the repository and can automatically redeploy a linked service when the configured branch changes.