#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_subscriber')

        # Criador da ponte ROS ↔ OpenCV
        self.bridge = CvBridge()

        # Inscrição no tópico da câmera
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # tópico vindo do Gazebo
            self.image_callback,
            10
        )
        self.subscription  # evita warning de variável não usada

    def image_callback(self, msg):
        try:
            # Converte imagem ROS para formato OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Mostra a imagem
            cv2.imshow("Câmera do Robô", cv_image)
            cv2.waitKey(1)

            # Exemplo de processamento (converter para cinza)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Exemplo de detecção de bordas
            edges = cv2.Canny(gray, 100, 200)
            cv2.imshow("Detecção de Bordas", edges)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Erro ao converter imagem: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CameraSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
