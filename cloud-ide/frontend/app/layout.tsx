import './globals.css'

export const metadata = {
  title: 'Orca Cloud IDE - Game Development in the Browser',
  description: 'Build games with Orca Engine directly in your browser. No downloads required.',
  icons: {
    icon: '/favicon.png',
    apple: '/favicon.png'
  }
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
