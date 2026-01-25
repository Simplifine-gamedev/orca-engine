import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Orca RTS - Wood Gathering',
  description: 'Real-time strategy game with resource gathering',
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
