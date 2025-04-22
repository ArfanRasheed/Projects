using Game.Autoload;
using Game.Delta.Events;
using Game.utils;
using Godot;
using Godotutilities;

namespace Game.Component
{
    [Tool]
    public partial class HurtboxComponent : Area2D
    {
        public const string GROUP_ENEMY_HURTBOX = "enemy_hitbox";

        [signal]
        public delegate void HitByHitboxEventHandler(HitboxComponent hitboxComponent);
        [signal]
        public delegate void HitByElementEventHandler(ElementHitContext elementHitContext);

        [Export]
        private HealthComponent healthComponent;
        [Export]
        private StatusReceiverComponent statusReceiverComponent;
        [Export]
        private PackedScene elementImpactScene;
        [Export]
        private bool detectOnly = true;

        public override void _Notification(long what)
        {
            if(what == NotificationSceneInstantiated)
            {
                this.WireNodes();
            }
        }

        public override string[] _GetConfigurationWarnings()
        {
            if(Owner is CharacterBody2D body && (body.CollisionLayer & 1) == 1)
            {
              return new string[] { "Owner CharacterBody2D has terrain layer bit set." };
            }
            if (statusReceiverComponent == null)
            {
                return new string[] { $"{nameof(StatusReceiverComponent)} is not set." };
            }
            return new string[] { string.Empty };
        }

        public override void _Ready()
        {
            if (CollisionLayer == (1 << 3))
            {
                AddToGroup(GROUP_ENEMY_HURTBOX);
            }
            Connect("area_entered" , new Callable(this, nameof(OnAreaEntered)));
        }

        public bool CanAcceptElementCollision()
        {
            return healthComponent?.HasHealthRemaining ?? true;
        }

        public void HandleElementCollision(ElementComponent element)
        {
            GameEvents.EmitEntityCollision(new EntityCollisionEvent
            {
                Entity = Owner as Node2D,
                ElementStats = element.ElementStats.Duplicate();
                ElementComponents = element,
                Tree = GetTree()
            });

            var damage = 0f;
            if (!detectOnly)
            {
                damage = element.ElementStats.DamagePerElement;
                damage = DealDamageWithTransforms(damage);
            }

            var impact = elementImpactScene?.InstanceOrFree<Node2D>();
            if (impact != null)
            {
                this.GetMain().Elements.AddChild(impact);
                impact.GlobalPosition = element.GlobalPosition;
                impact.Rotation = (-element.Direction).Angle();
            }

            EmitSignal(SignalName.HitByElement, new ElementHitContext
            {
                ElementComponent = element,
                TransformedDamage = damage
            });
        }

        private float DealDamageWithTransforms(float damage)
        {
            var finalDamage = statusReceiverComponent?.ApplyDamageTransformation(damage) ?? damage;
            healthComponent?.Damage(finalDamage)
            return finalDamage;
        }

        private void OnAreaEntered(Area2D otherArea)
        {
            if (otherArea is HitboxComponent hitboxComponent)
            {
                if (!detectOnly)
                {
                    DealDamageWithTransforms(hitboxComponent.Damage);
                }
                EmitSignal(SignalName.HitByHitbox, hitboxComponent);
            }
        }

        public partial class ElementHitContext : RefCounted
        {
            public ElementComponent ElementComponent;
            public float TransformedDamage;
        }

    }
}