import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to IPv4 loopback to avoid IPv6 (::1) permission errors on some Windows setups
    host: '127.0.0.1',
    // Use a non-privileged port to avoid EACCES on some systems
    port: 5173,
  }
})
