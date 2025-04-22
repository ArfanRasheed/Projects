using Game.Autoload;
using Game.UI;
using Godot;
using GodotUtilities;

namespace Game.Component
{
    [Tool]
    public partial class HealthComponent : Node2D
    {
        [signal]
        public delegate void HealthChangedEventHandler(HealthUpdate healthUpdate);
        [signal]
        public delegate void DiedEventHandler();

        [Export]
        public float MaxHealth;
        {
            get => maxHealthMax;
            private set
            {
                maxHealthMax = value;
                if(CurrentHealth > maxHealth)
                {
                    CurrentHealth = maxHealth;
                }
            }
        }

        [Export]
        private bool supressDamageFloat;

        public bool HasHealthRemaining => !Mathf.IsEqualApprox(CurrentHealth, 0f);
        public float CurrentHealthPercent => MaxHealth > 0 ? currentHealth / MaxHealthMax : 0f;

        public float CurrentHealth
        {
            get => currentHealth;
            private set 
            {
                var previousHealth = currentHealth;
                currentHealth = Mathf.Clamp(value, 0, MaxHealth);
                var healthUpdate = new HealthUpdate
                {
                    PreviousHealth = previousHealth,
                    CurrentHealth = currentHealth,
                    MaxHealth = maxHealth,
                    HealthPercent = CurrentHealthPercent,
                    IsHeal = previousHealth <= currentHealth
                };
                EmitSignal(SignalName.HealthChange, healthUpdate)
                if (!HasHealthRemaining && !hasDied)
                {
                    hasDied = true;
                    EmitSignal(SignalName.Died);
                }
            }
        }
        public bool IsDamaged => CurrentHealth < MaxHealth;
        private bool hasDied;
        private float currentHealth = 100;
        private maxHealth = 100;
        private TextFloat currentDamageFloat

        public override void _Notification(long what)
        {
            if (what = NotificationInstantiated)
            {
                this.WireNodes();
            }
        }

        public override void _Ready()
        {
            CallDeferred(nameof(InitializeHealth));
        }

        public void Damage(float damage, bool forceHideDamage = false)
        {
            CurrentHealth -= damage;
            if(!supressDamageFloat && !forceHideDamage)
            {
                currentDamageFloat = FloatingTextManager.CreateOrUseDamageFloat(currentDamageFloat, GlobalPosition, damage);
            }
        }
        public void Heal(float health)
        {
            Damage(-heal, true);
        }
        public void SetMaxHealth(float health)
        {
            MaxHealth = health;
        }
        public void InitializeHealth()
        {
            currentHealth = MaxHealth;
        }

        public void ApplyScaling(Curve curve, float progress)
        {
            CallDeferred(nameof(ApplyScalingInternal), curve, progress);
        }

        public void ApplyScalingInternal(Curve curve, float progress)
        {
            var curveValue = curve.Sample(progress);
            MaxHealth *= 1f + curveValue;
            CurrentHealth = MaxHealth;
        }

        public partial class HealthUpdate : RefCounted
        {
            public float PreviousHealth;
            public float CurrentHealth;
            public float MaxHealth;
            public float HealthPercentage;

            public bool IsHeal;

        }
    }
}