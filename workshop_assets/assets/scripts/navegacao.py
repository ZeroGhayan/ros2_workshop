import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import math

class ZigZagNavigator(Node):
    def __init__(self, linhas, colunas, espacamento):
        super().__init__('zigzag_navigator')
        self.linhas = linhas
        self.colunas = colunas
        self.espacamento = espacamento

        self.path = self.generate_zigzag_path()
        self.current_target_idx = 0
        self.current_pose = None

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscriber = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)

	# Parâmetros da navegação
	self.declare_parameter('tolerance', 0.25)
	self.declare_parameter('linear_speed', 0.3)
	self.declare_parameter('angular_gain', 1.0)

        # Câmera
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/camera', self.image_callback, 10)
        self.video_writer = cv2.VideoWriter(
            'output_video.mp4',
            cv2.VideoWriter_fourcc(*'mp4v'),
            10, (800, 600)
        )

    def generate_zigzag_path(self):
        path = []
        for i in range(self.linhas):
            row = range(self.colunas) if i % 2 == 0 else reversed(range(self.colunas))
            for j in row:
                x = j * self.espacamento
                y = i * self.espacamento
                path.append((x, y))
        return path

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
		        self.get_logger().info("🏁 Caminho completo! Parando robô.")
		        self.publisher.publish(Twist())  # Para o robô
		        self.video_writer.release()
		        cv2.destroyAllWindows()
		    return

		# Posição atual
		x = self.current_pose.position.x
		y = self.current_pose.position.y
		yaw = self.get_yaw()

		# Alvo atual
		target_x, target_y = self.path[self.current_target_idx]

		# Distância ao alvo atual
		dist_to_target = self.distance(x, y, target_x, target_y)

		# Normalizar diferença de ângulo (shortest path)
		angle_to_target = math.atan2(target_y - y, target_x - x)
		angle_diff = angle_to_target - yaw
		# Wrap para [-pi, pi]
		while angle_diff > math.pi:
		    angle_diff -= 2 * math.pi
		while angle_diff < -math.pi:
		    angle_diff += 2 * math.pi

		twist = Twist()

		# Se estiver muito perto do waypoint, passa pro próximo
		if dist_to_target < 0.25:  # Aumentei um pouco a tolerância (ajuste se preciso)
		    self.get_logger().info(f"✅ Waypoint {self.current_target_idx + 1}/{len(self.path)} atingido")
		    self.current_target_idx += 1
		    self.publisher.publish(Twist())  # Para brevemente
		    return

		# Controle proporcional
		K_angular = 1.0   # Ganho angular (ajuste: mais alto = gira mais rápido)
		K_linear = 0.3     # Ganho linear (vel máxima ~0.3 m/s)

		max_angular = 0.8  # Limite segurança
		max_linear = 0.3

		# Velocidade angular proporcional ao erro
		twist.angular.z = max(min(K_angular * angle_diff, max_angular), -max_angular)

		# Velocidade linear: reduz quando precisa girar muito
		if abs(angle_diff) < 0.3:  # ~17° - quase alinhado
		    twist.linear.x = min(K_linear, max_linear)
		    # Opcional: linear proporcional à distância (mais lento perto do alvo)
		    # twist.linear.x = min(K_linear * dist_to_target, max_linear)
		else:
		    twist.linear.x = 0.0  # Só gira no lugar se erro grande

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
    navigator = ZigZagNavigator(linhas, colunas, espacamento)

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
