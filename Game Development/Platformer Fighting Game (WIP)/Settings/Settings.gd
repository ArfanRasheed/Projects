extends Node2D

@onready var option_button1 = $OptionButton1 as OptionButton
@onready var option_button2 = $OptionButton2 as OptionButton
# Called when the node enters the scene tree for the first time.
func _ready():
	#$HSlider.value = Globals.volume1
	$HSlider2.value = Globals.volume2
	$HSlider3.value = Globals.volume3
	$HSlider4.value = Globals.volume4
	add_window_mode_items()
	add_resolution_items()
	option_button1.item_selected.connect(on_window_mode_selected)
	option_button2.item_selected.connect(on_resolution_selected)
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass





const WINDOW_MODE_ARRAY : Array[String] = [
	"Full-screen",
	"Window Mode",
	"Borderless Full-screen"
]

const RESOLUTION_DICTIONARY : Dictionary = {
	"1152 x 648" : Vector2(1152, 648),
	"1280 x 720" : Vector2(1280, 720),
	"1920 x 1080" : Vector2(1920, 1080)
}
func add_window_mode_items() -> void:
	for window_mode in WINDOW_MODE_ARRAY:
		option_button1.add_item(window_mode)
	

func on_window_mode_selected(index : int) -> void:
	match index:
		0: #Full screen
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
			DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, false)
		1: #Window Mode
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
			DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, false)	
		2: #Full screen
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
			DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, true)

func add_resolution_items() -> void:
	for resolution_size_text in RESOLUTION_DICTIONARY:
		option_button2.add_item(resolution_size_text)
	
func on_resolution_selected(index : int) -> void:
	DisplayServer.window_set_size(RESOLUTION_DICTIONARY.values()[index])


func _on_back_setting_pressed():
	get_tree().change_scene_to_file("res://MainMenu/MainMenu.tscn")






