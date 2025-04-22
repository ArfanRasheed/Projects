extends CharacterBody2D


@export var speed : float = 400.0
@export var jump_velocity : float = -400.0
@export var double_jump_velocity : float = -400.0
@onready var animated_sprite : AnimatedSprite2D = $AnimatedSprite2D
@export var explosion : PackedScene
# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")
var has_double_jumped : bool = false
var animation_locked : bool = false
var direction : Vector2 = Vector2.ZERO
var was_in_air : bool = false
func _physics_process(delta):
	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta
		was_in_air = true
	else:
		has_double_jumped = false
		if was_in_air == true:
			land()
		was_in_air = false
	# Handle Jump.
	if Input.is_action_just_pressed("jump"): 
		if is_on_floor():
			#normal jump
			jump()
			
		elif not has_double_jumped:
			#double jump
			velocity.y = double_jump_velocity
			has_double_jumped = true

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	direction = Input.get_vector("left", "right","up","down")
	if direction.x != 0 && animated_sprite.animation != "JumpDown":
		velocity.x = direction.x * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
	
	move_and_slide()
	update_animation()
	update_facing_direction()
	right_attack()
	left_attack()
	up_attack()
	
func update_animation():
	if not animation_locked:
		if not is_on_floor():
			animated_sprite.play("JumpDown")
		if direction.x != 0:
			animated_sprite.play("Run")
		else:
			animated_sprite.play("Idle")
			

func update_facing_direction():
	if direction.x > 0:
		animated_sprite.flip_h = false
	elif direction.x < 0:
		animated_sprite.flip_h = true
		
func jump():
	velocity.y = jump_velocity
	animated_sprite.play("JumpUp")
	animation_locked = true

func land():
	animated_sprite.play("JumpDown")
	animation_locked = true



func _on_animated_sprite_2d_animation_finished():
	if(animated_sprite.animation == "JumpDown"):
		animation_locked = false
	elif(animated_sprite.animation == "JumpUp"):
		animation_locked = false
	elif(animated_sprite.animation == "Attack"):
		animation_locked = false


func right_attack():
	if Input.is_action_just_pressed("Attack1"):  
		animated_sprite.play("Attack")
		$AttackSprite/AnimationPlayer.play("rightattack")
		animation_locked = true

func left_attack():
	if Input.is_action_just_pressed("Attack2"): 
		animated_sprite.play("Attack")
		$AttackSprite/AnimationPlayer.play("leftattack")
		animation_locked = true

func up_attack():
	if Input.is_action_just_pressed("Attack3"): 
		animated_sprite.play("Attack")
		$AttackSprite/AnimationPlayer.play("upattack")
		animation_locked = true
