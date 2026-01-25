import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Orca RTS - Rally Point to Resource',
  description: 'Real-time strategy game with automatic worker gathering from rally points',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
