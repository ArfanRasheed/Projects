extends Node

var player 
var playerScript = preload("res://CharacterMovement/player.gd")

var opponent
var aiScript = preload("res://CharacterMovement/player.gd")

var selectableCharacters = {
	"Leaf Ranger" : preload("res://CharacterSelect/LeafRanger.tscn"),
	"Crystal Mauler" : preload("res://CharacterSelect/CrystalMauler.tscn"),
	"Fire Knight" : preload("res://CharacterSelect/FireKnight.tscn"),
	"Wind Hashira" : preload("res://CharacterSelect/WindHashira.tscn"),
	"Water Priestess" : preload("res://CharacterSelect/WaterPriestess.tscn"),
	"Metal Bladesmith" : preload("res://CharacterSelect/MetalBladesmith.tscn"),
	"Monk" : preload("res://CharacterSelect/Monk.tscn"),
	"Wizard" : preload("res://CharacterSelect/Wizard.tscn"),
	
}
