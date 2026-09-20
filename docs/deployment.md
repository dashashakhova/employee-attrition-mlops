# Deployment and verification

## Local

Start the complete stack:

```bash
docker compose up --build -d
docker compose ps
```

Expected core endpoints:

- FastAPI Swagger: `http://localhost:8000/docs`
- FastAPI health: `http://localhost:8000/health`
- FastAPI metrics: `http://localhost:8000/metrics`
- MLflow UI: `http://localhost:8080`
- Airflow UI: `http://localhost:8081`
- MinIO console: `http://localhost:9001`

## Cloud

`render.yaml` declares the public FastAPI service on Render. The API binds to `0.0.0.0` and uses Render's `PORT` environment variable.

After Render deploys the `employee-attrition-api` service, verify:

```text
GET /health -> HTTP 200
GET /docs -> Swagger UI
POST /predict -> JSON prediction
```

Render health checks target `/health`.

## CI/CD

GitHub Actions validates tests and the API Docker build on pushes and pull requests. Render `autoDeploy: true` redeploys the linked service when the configured branch changes. A `RENDER_DEPLOY_HOOK` GitHub secret can also be used for an explicit deployment hook.