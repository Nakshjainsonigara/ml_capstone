output "container_id" {
  description = "ID of the running FastAPI container"
  value       = docker_container.api.id
}

output "endpoint" {
  description = "URL to reach the FastAPI service"
  value       = "http://localhost:${var.host_port}"
}
