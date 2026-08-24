# Frontend autonome pour déploiement distant (K8s ou compose).
# Sert le SPA React en statique ET proxie /api/ vers le backend de <backend-host>.
# Contexte de build attendu : racine du dépôt.
#   docker build -f deploy/frontend/Dockerfile -t junon-frontend .

FROM node:20-alpine AS builder
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# Script bridge de l'observatoire (servi en statique, cf. nginx.conf.template)
COPY nginx/observatory-bridge.js /usr/share/nginx/html/observatory-bridge.js
# Template traité par envsubst au démarrage : ${DIB_BACKEND} y est substitué.
COPY deploy/frontend/nginx.conf.template /etc/nginx/templates/default.conf.template
EXPOSE 80
# L'entrypoint officiel nginx applique envsubst sur /etc/nginx/templates/*.template
# puis lance nginx. Aucun CMD custom nécessaire.
