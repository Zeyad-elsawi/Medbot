# Stage 1: Build the React application
FROM node:18-alpine AS build

WORKDIR /app

# Install dependencies
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the application
COPY . .

# Build the Vite app for production
RUN npm run build

# Stage 2: Serve the application with Nginx
FROM nginx:alpine

# Remove default Nginx website
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from the build stage
COPY --from=build /app/dist /usr/share/nginx/html

# Copy custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
