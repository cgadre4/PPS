import pygame

# Window settings
WIDTH, HEIGHT = 800, 500
CAPTION = "Episode 1 — Step 1: Draw One Particle"

# Particle settings
PARTICLE_X, PARTICLE_Y = 400, 250
RADIUS = 10

BG_COLOR = (15, 15, 20)
PARTICLE_COLOR = (220, 220, 240)

def main():
    pygame.init()

    # Create window (a "surface" we can draw onto)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(CAPTION)

    # 1) Fill background once
    screen.fill(BG_COLOR)

    # 2) Draw one circle once
    pygame.draw.circle(screen, PARTICLE_COLOR, (PARTICLE_X, PARTICLE_Y), RADIUS)

    # 3) Update the display so the drawing appears
    pygame.display.flip()

    # 4) Keep the window open until the user closes it
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
