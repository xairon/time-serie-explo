FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# Templated config: nginx:alpine entrypoint runs envsubst on /etc/nginx/templates/*.template
# so the backend URL can be set per environment (dib compose vs K8s) without rebuilding.
COPY docker/frontend/nginx.conf /etc/nginx/templates/default.conf.template
# Default target = docker-compose service on dib. On K8s, override with
# BACKEND_URL=http://10.195.25.16:49514 in the deployment env.
ENV BACKEND_URL=http://backend:8000
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
