extends Node2D

var PlayerSelect = 0 

func _process(delta):
	match PlayerSelect:
		0:
			get_node("PlayerSelect").play("Player0")
			get_node("PlayerSelect/Descript").text = "Name : Fire Knight \nAbilities : "
		1:
			get_node("PlayerSelect").play("Player1")
			get_node("PlayerSelect/Descript").text = "Name : Leaf Ranger \nAbilities : "
		2:
			get_node("PlayerSelect").play("Player2")
			get_node("PlayerSelect/Descript").text = "Name : Metal Bladeskeeper \nAbilities : "
		3:
			get_node("PlayerSelect").play("Player3")
			get_node("PlayerSelect/Descript").text = "Name : Air Monk \nAbilities : "
		4:
			get_node("PlayerSelect").play("Player4")
			get_node("PlayerSelect/Descript").text = "Name : Water Priestess \nAbilities : "
		5:
			get_node("PlayerSelect").play("Player5")
			get_node("PlayerSelect/Descript").text = "Name : Wind Hashimara \nAbilities : "
		
func _on_left_pressed():
	if PlayerSelect >= 1:
		PlayerSelect -= 1
	elif PlayerSelect == 0:
		PlayerSelect = 6

func _on_right_pressed():
	if PlayerSelect < 5:
		PlayerSelect += 1
	elif PlayerSelect == 5:
		PlayerSelect = 0


func _on_ready_pressed():
	get_tree().change_scene_to_file("res://MapSelection/MapSelection.tscn")


func _on_back_pressed():
	get_tree().change_scene_to_file("res://MainMenu/MainMenu.tscn")
