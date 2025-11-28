FROM node:18-alpine

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source code
COPY . .

# Build app
RUN npm run build

# Expose port
EXPOSE 3000

# Run development server (for dev) or production server
CMD ["npm", "run", "dev"]