import { resolve } from 'path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin({
      exclude: [
        'libp2p',
        '@libp2p/tcp',
        '@libp2p/kad-dht',
        '@libp2p/identify',
        '@libp2p/ping',
        '@libp2p/peer-id',
        '@libp2p/interface',
        '@chainsafe/libp2p-noise',
        '@chainsafe/libp2p-yamux',
        'multiformats',
        '@multiformats/multiaddr',
      ],
    })]
  },
  preload: {
    plugins: [externalizeDepsPlugin()]
  },
  renderer: {
    resolve: {
      alias: {
        '@renderer': resolve('src/renderer/src')
      }
    },
    plugins: [react()]
  }
})
