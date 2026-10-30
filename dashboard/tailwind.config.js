/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neon-cyan':    '#00f0ff',
        'neon-magenta': '#ff00d9',
        'neon-green':   '#00ff8c',
        'neon-yellow':  '#ffe600',
        'neon-red':     '#ff0044',
        'neon-purple':  '#a855f7',
        'bg-deep':      '#050810',
        'bg-base':      '#0a0e1a',
        'bg-card':      '#0f1420',
        'bg-elevated':  '#161b2a',
      },
      fontFamily: {
        orbitron: ['Orbitron', 'sans-serif'],
        mono:     ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'scan-line':  'scan-line 4s linear infinite',
        'float':      'float 4s ease-in-out infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1',   filter: 'brightness(1)' },
          '50%':       { opacity: '0.6', filter: 'brightness(1.4)' },
        },
        'scan-line': {
          '0%':   { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%':       { transform: 'translateY(-8px)' },
        },
      },
      boxShadow: {
        'neon-cyan':    '0 0 8px #00f0ff, 0 0 24px #00f0ff40',
        'neon-magenta': '0 0 8px #ff00d9, 0 0 24px #ff00d940',
        'neon-green':   '0 0 8px #00ff8c, 0 0 24px #00ff8c40',
        'neon-red':     '0 0 8px #ff0044, 0 0 24px #ff004440',
        'neon-purple':  '0 0 8px #a855f7, 0 0 24px #a855f740',
      },
    },
  },
  plugins: [],
}
