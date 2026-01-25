/**
 * Example: Connecting client to server via Socket.IO
 */

const io = require('socket.io-client');

// Connect to the game server
const socket = io('http://localhost:3001');

// Connection events
socket.on('connect', () => {
  console.log('Connected to game server');
  
  // Request initial game state
  socket.on('game:state', (state) => {
    console.log('Initial game state:', state);
  });
});

socket.on('disconnect', () => {
  console.log('Disconnected from game server');
});

// Lair events
socket.on('lair:created', (lair) => {
  console.log('Lair created:', lair);
});

socket.on('lair:damaged', (data) => {
  console.log(`Lair ${data.lairId} damaged. Health: ${data.health}`);
  if (data.destroyed) {
    console.log('Lair destroyed! Loot:', data.loot);
  }
});

socket.on('lair:destroyed', (data) => {
  console.log(`Lair ${data.lairId} destroyed. Loot:`, data.loot);
});

// Mob events
socket.on('mob:spawned', (mob) => {
  console.log(`Mob spawned: ${mob.type} (Lv ${mob.level}) from lair ${mob.lairId}`);
});

socket.on('mob:killed', (data) => {
  console.log(`Mob ${data.mobId} killed`);
});

socket.on('mob:damaged', (data) => {
  console.log(`Mob ${data.mobId} damaged. Health: ${data.health}`);
  if (data.killed) {
    console.log('Mob killed!');
  }
});

// Example: Create a lair
setTimeout(() => {
  console.log('Creating a goblin camp...');
  socket.emit('lair:create', {
    type: 'goblin_camp',
    position: { x: 100, y: 100 }
  });
}, 1000);

// Example: Attack a lair after 5 seconds
setTimeout(() => {
  // Note: You'd need to track the lairId from the lair:created event
  // This is just a demonstration
  console.log('Attacking lair...');
  socket.emit('lair:damage', {
    lairId: 'replace_with_actual_lair_id',
    damage: 100
  });
}, 5000);

// Keep the script running
process.stdin.resume();

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\nDisconnecting...');
  socket.disconnect();
  process.exit(0);
});
