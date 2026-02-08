ARG FRONTEND_NODE_IMAGE=node:22-alpine
FROM ${FRONTEND_NODE_IMAGE} AS build

WORKDIR /app
COPY package*.json ./
ARG NPM_REGISTRY=https://registry.npmmirror.com
RUN npm config set registry ${NPM_REGISTRY} \
    && npm ci --prefer-offline --no-audit --fund=false

COPY . .
RUN npm run build

FROM nginx:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
