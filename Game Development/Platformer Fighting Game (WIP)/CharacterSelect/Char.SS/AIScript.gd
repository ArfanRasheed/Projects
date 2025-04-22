extends Node


@onready var gameRef = get_parent()
@onready var player = get_parent().get_node(gameRef.givenPlayer.name)  # Target to focus on
