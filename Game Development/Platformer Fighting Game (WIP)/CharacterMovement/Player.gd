extends CharacterBody2D


@export var speed : float = 200.0
@export var jump_velocity : float = -150.0
@export var double_jump_velocity : float = -135.0

@onready var animated_sprite : AnimatedSprite2D = $AnimatedSprite2D

# Get the gravity from the project settings to be synced with RigidBody nodes.
var gravity = ProjectSettings.get_setting("physics/2d/default_gravity")
var has_double_jumped : bool = false
var animation_locked : bool = false
var direction : Vector2 = Vector2.ZERO  
var was_in_ar : bool = false 


func _physics_process(delta):
	# Add the gravity.
	if not is_on_floor():
		velocity.y += gravity * delta
		was_in_ar = false     ###### LOOK AT ME 
	else:
		has_double_jumped = false
		
		if was_in_ar == true:
			land()
		was_in_ar = false
	
	# Handle Jump.
	if Input.is_action_just_pressed("jump"):
		if is_on_floor():
			# Normal Jump on floor
			jump()
		elif not has_double_jumped:
			# double jump in air
			double_jump()
			

	# Get the input direction and handle the movement/deceleration.
	# As good practice, you should replace UI actions with custom gameplay actions.
	direction = Input.get_vector("left", "right", "up", "down")
	
	
	if direction.x != 0 && animated_sprite.animation != "jump_end":
		velocity.x = direction.x * speed
	else:
		velocity.x = move_toward(velocity.x, 0, speed)
	update_animation()
	update_facing_direction()
	move_and_slide()

func update_animation():
	if not animation_locked:
		if not is_on_floor():
			animated_sprite.play("jump_loop")
			
		else:
			if direction.x != 0:
				animated_sprite.play("run")
			else:
				animated_sprite.play("idle")
	
func update_facing_direction(): 
	if direction.x > 0:
		animated_sprite.flip_h = false  # value is flipped in the right direction
	elif direction.x < 0:
		animated_sprite.flip_h = true  # flip left direction
	
func jump():
	velocity.y = jump_velocity
	animated_sprite.play("jump_start")
	animation_locked = true
	
func double_jump():
	velocity.y = double_jump_velocity
	animated_sprite.play("jump_double")
	animation_locked = true
	has_double_jumped = true

func land():
	animated_sprite.play("jump_end")
	animation_locked = true
	
func attack1():
	animated_sprite.play("attack1")
	animation_locked = true
	
func attack2():
	animated_sprite.play("attack2")
	animation_locked = true

func knockback():
	animated_sprite.play("knockback")
	animation_locked = true
	
func defend():
	animated_sprite.play("defend")
	animation_locked = true
	
func roll():
	animated_sprite.play("roll")
	animation_locked = true
	
func special():
	animated_sprite.play("special")
	animation_locked = true


func _on_animated_sprite_2d_animation_finished():
	if(["jump_end", "jump_start", "jump_double"].has (animated_sprite.animation)):
		animation_locked = false
