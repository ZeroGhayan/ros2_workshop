import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math
import heapq
import random  # Para ruído probabilístico

class AStarNavigator(Node):
    def __init__(self, linhas, colunas, espacamento):
        super().__init__('astar_navigator')
        self.linhas = linhas
        self.colunas = colunas
        self.espacamento = espacamento

        # Defina start, goal e obstáculos (em coords odom: start= (0,0))
        start = (0, 0)  # Início no odom
        goal = ((colunas - 1) * espacamento, (linhas - 1) * espacamento)  # Fim da grade
        obstacles = [(8, 0)]  # Caixa em odom ~ (8,0) - adicione mais se precisar [(x1,y1), (x2,y2), ...]

        self.path = self.generate_astar_path(start, goal, obstacles)
        if not self.path:
            self.get_logger().error("❌ Nenhum caminho encontrado com A*! Verifique obstáculos.")
            return

        self.current_target_idx = 0
        self.current_pose = None

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

        # Câmera (mantido igual)
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.video_writer = cv2.VideoWriter(
            'output_video.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (800, 600)
        )

    def generate_astar_path(self, start, goal, obstacles):
        cell_size = 1.0  # Resolução fina pro A* (pra desviar bem)
        grid_min_x, grid_min_y = min(start[0], goal[0]) - 5, min(start[1], goal[1]) - 5
        grid_max_x, grid_max_y = max(start[0], goal[0]) + 5, max(start[1], goal[1]) + 5
        grid_width = int((grid_max_x - grid_min_x) / cell_size)
        grid_height = int((grid_max_y - grid_min_y) / cell_size)

        # Grid de obstáculos (set pra rápido lookup)
        obs_set = set()
        for ox, oy in obstacles:
            gx = int((ox - grid_min_x) / cell_size)
            gy = int((oy - grid_min_y) / cell_size)
            obs_set.add((gx, gy))
            # Expanda obstáculo pra segurança (buffer 1 célula)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    obs_set.add((gx + dx, gy + dy))

        start_grid = (int((start[0] - grid_min_x) / cell_size), int((start[1] - grid_min_y) / cell_size))
        goal_grid = (int((goal[0] - grid_min_x) / cell_size), int((goal[1] - grid_min_y) / cell_size))

        neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]  # Diagonais pra caminhos suaves

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: math.dist(start_grid, goal_grid)}  # Heurística Euclidiana

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # Reconstrói caminho
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                # Converte de volta pra coords mundo (centro da célula)
                return [(grid_min_x + p[0] * cell_size + cell_size / 2,
                         grid_min_y + p[1] * cell_size + cell_size / 2) for p in path]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if not (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height):
                    continue
                if neighbor in obs_set:
                    continue

                # Custo com ruído probabilístico (gaussiano, pra robótica prob)
                cost = math.dist(current, neighbor) + random.gauss(0, 0.1)  # Mu=0, sigma=0.1

                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + math.dist(neighbor, goal_grid)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # Sem caminho

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def get_yaw(self):
        orientation = self.current_pose.orientation
        qx, qy, qz, qw = orientation.x, orientation.y, orientation.z, orientation.w
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        return math.atan2(siny_cosp, cosy_cosp)

    def distance(self, x1, y1, x2, y2):
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def timer_callback(self):
        if self.current_pose is None or self.current_target_idx >= len(self.path):
            if self.current_target_idx >= len(self.path):
                self.get_logger().info("🏁 Labirinto completo! Parando robô.")
                self.publisher.publish(Twist())  # Para
                self.video_writer.release()
                cv2.destroyAllWindows()
            return

        x = self.current_pose.position.x
        y = self.current_pose.position.y
        yaw = self.get_yaw()

        target_x, target_y = self.path[self.current_target_idx]
        dist_to_target = self.distance(x, y, target_x, target_y)

        angle_to_target = math.atan2(target_y - y, target_x - x)
        angle_diff = angle_to_target - yaw
        # Wrap [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        twist = Twist()

        if dist_to_target < 0.25:  # Tolerância
            self.get_logger().info(f"✅ Waypoint {self.current_target_idx + 1}/{len(self.path)} atingido")
            self.current_target_idx += 1
            self.publisher.publish(Twist())  # Para breve
            return

        # Ganho proporcional
        K_angular = 1.0
        K_linear = 0.3
        max_angular = 0.8
        max_linear = 0.3

        twist.angular.z = max(min(K_angular * angle_diff, max_angular), -max_angular)

        if abs(angle_diff) < 0.3:
            twist.linear.x = min(K_linear * dist_to_target, max_linear)  # Proporcional à dist
        else:
            twist.linear.x = 0.0

        self.publisher.publish(twist)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            resized_image = cv2.resize(cv_image, (800, 600))
            cv2.imshow("Camera View", resized_image)
            self.video_writer.write(resized_image)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Erro ao processar imagem: {e}")

def main():
    rclpy.init()
    linhas = 4
    colunas = 10
    espacamento = 2.0
    navigator = AStarNavigator(linhas, colunas, espacamento)

    try:
        rclpy.spin(navigator)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
