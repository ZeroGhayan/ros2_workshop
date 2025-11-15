#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Visao(Node):
    def __init__(self):
        super().__init__('visao')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        self.frame = None
        self.get_logger().info("Nó de visão iniciado. Aguardando imagens...")

        # Cria uma janela única e fixa
        cv2.namedWindow("Câmera (Contornos)", cv2.WINDOW_NORMAL)

    def image_callback(self, msg):
        # Atualiza apenas o frame atual
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def processar_e_mostrar(self):
        if self.frame is None:
            return True  # ainda não recebeu imagem

        # Converte para cinza e aplica blur
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detecção de bordas e contornos
        edges = cv2.Canny(blurred, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoured = self.frame.copy()
        cv2.drawContours(contoured, contours, -1, (255, 0, 0), 2)

        # Mostra na janela única
        cv2.imshow("Câmera (Contornos)", contoured)

        # Fecha com 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        return True


def main(args=None):
    rclpy.init(args=args)
    node = Visao()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if not node.processar_e_mostrar():
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info("Encerrando nó de visão.")
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

