import pygame
import queen

SQUARE_SIZE = 50
GRID_SIZE = 8
QUEEN = pygame.transform.scale(pygame.image.load('8-Queens\images\chess-queen.svg'), (SQUARE_SIZE, SQUARE_SIZE))
queen_width, queen_height = QUEEN.get_size()

WHITE = (255, 255, 255)
BROWN = (139, 69, 19)

pygame.init()
screen = pygame.display.set_mode((GRID_SIZE * SQUARE_SIZE, GRID_SIZE * SQUARE_SIZE))

for x in range(GRID_SIZE):
    for y in range(GRID_SIZE):
        rect = pygame.Rect(x*SQUARE_SIZE, y*SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
        pygame.draw.rect(screen, WHITE if (x+y) % 2 == 0 else BROWN, rect)

pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if pygame.key.get_pressed()[pygame.K_SPACE]:
         
        solver = queen.Eight_Queens()

        for x, y in solver.solve():
            screen.blit(QUEEN, (x*SQUARE_SIZE + 6, y*SQUARE_SIZE + 5))
        pygame.display.flip()

pygame.quit()