/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#ffffff',
        surface: '#f9fafb',
        border: '#e5e7eb',
        accent: '#2563eb',
        'accent-strong': '#1d4ed8',
        text: '#111827',
        muted: '#6b7280',
      },
    },
  },
  plugins: [],
}
