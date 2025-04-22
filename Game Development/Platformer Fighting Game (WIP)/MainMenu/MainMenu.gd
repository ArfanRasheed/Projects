extends Control

@export var Address = "127.0.0.1"
@export var port = 8910
var peer
# Called when the node enters the scene tree for the first time.
func _ready():
	#Globals.volume2 = $HSlider2.value
	multiplayer.peer_connected.connect(peer_connected)
	multiplayer.peer_disconnected.connect(peer_disconnected)
	multiplayer.connected_to_server.connect(connected_to_server)
	multiplayer.connected_failed.connect(connection_failed)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass

func peer_connected(id):
	print("Player Connected " + str(id))

func peer_disconnected(id):
	print("Player Disconnected " + str(id))

func connected_to_server():
	print("Connected to server")
	SendPlayerInformation.rpc_id(1, multiplayer.get_unique_id())
	
func connection_failed():
	print("Connection failed")

@rpc("any_peer","call_local")
func SendPlayerInformation(id):
	if !GameManager.Players.has(id):
		GameManager.Players[id] = {
			"id" = id
		}
	if multiplayer.is_server():
		for i in GameManager.Players:
			SendPlayerInformation.rpc(GameManager.Players[i].id, i)
@rpc("any_peer", "call_local")
func StartGame():
	get_tree().change_scene_to_file("res://CharacterSelect/CharacterSelect.tscn")


func _on_settings_button_pressed():
	get_tree().change_scene_to_file("res://Settings/Settings.tscn")


func _on_discord_pressed():
	OS.shell_open("https://discord.gg/DFmnsfPZCx")


func _on_host_button_pressed():
	peer = ENetMultiplayerPeer.new()
	var error = peer.create_server(port, 2)
	if error != OK:
		print("Cannot host: " + error)
		return
	peer.get_host().compress(ENetConnection.COMPRESS_RANGE_CODER)
	
	multiplayer.set_multiplayer_peer(peer)
	print("Waiting For Players!")
	SendPlayerInformation(multiplayer.get_unique_id())

func _on_join_pressed():
	peer = ENetMultiplayerPeer.new()
	peer.create_client(Address, port)
	peer.get_host().compress(ENetConnection.COMPRESS_RANGE_CODER)
	multiplayer.set_multiplayer_peer(peer)


func _on_play_pressed():
	StartGame.rpc()
