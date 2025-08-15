#pragma once

#include "core/io/stream_peer_tcp.h"
#include "core/io/tcp_server.h"
#include "core/os/thread.h"
#include "core/os/mutex.h"

class AIToolServer : public RefCounted {
	GDCLASS(AIToolServer, RefCounted);

private:
	Ref<TCPServer> server;
	Ref<StreamPeerTCP> tcp;
	Ref<StreamPeer> peer;
	uint8_t req_buf[4096];
	int req_pos = 0;
	uint64_t time = 0;
	
	SafeFlag server_quit;
	Mutex server_lock;
	Thread server_thread;
	
	void _clear_client();
	void _send_response();
	void _poll();
	Dictionary _handle_tool_request(const String &p_method, const String &p_path, const String &p_body);
	
	static void _server_thread_poll(void *data);

protected:
	static void _bind_methods();

public:
	AIToolServer();
	~AIToolServer();
	
	void stop();
	Error listen(int p_port = 8001);
	bool is_listening() const;
}; 