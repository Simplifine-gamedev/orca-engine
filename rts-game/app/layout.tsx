import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Orca RTS - Blacksmith Building',
  description: 'Real-time strategy game with research and building mechanics',
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
