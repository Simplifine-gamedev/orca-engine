'use client'

import { useEffect } from 'react'
import { useTreeStore } from '../store/treeStore'

export default function TreeRegrowth() {
  const updateTrees = useTreeStore(state => state.updateTrees)

  useEffect(() => {
    const interval = setInterval(() => {
      updateTrees()
    }, 1000) // Check for regrowth every second

    return () => clearInterval(interval)
  }, [updateTrees])

  return null
}
