FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/package*.json ./
RUN npm ci --only=production
# Create public directory if it doesn't exist
RUN mkdir -p public
# Copy public folder if it exists
COPY --from=builder /app/public* ./public/
EXPOSE 3000
CMD ["npm", "start"]