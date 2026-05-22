FROM public.ecr.aws/docker/library/python:3.14-slim

WORKDIR /app

# Copy bundled package
COPY . /app

EXPOSE 8080

ENV PYTHONPATH=/app
ENV PATH="/app/bin:${PATH}"

# Auto-instrument with AWS Distro for OpenTelemetry
# https://aws-otel.github.io/docs/getting-started/python-sdk/auto-instr
CMD ["python", "bin/opentelemetry-instrument", "python", "-m", "my_project_ar4_robot_agent.agent.main"]
