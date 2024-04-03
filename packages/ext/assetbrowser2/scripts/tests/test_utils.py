# -*- coding: utf-8 -*-
import sys
sys.path.append("/Users/rndvfx/Documents/eclipseProjects/AssetBrowser/src")

from unittest import TestCase, main
try:
    from mock import create_autospec, patch
except ModuleNotFoundError:
    from unittest.mock import create_autospec, patch

from libs.utils import get_grid_size
from libs.utils import get_posix_file

class TestUtils(TestCase):

    @patch("libs.utils.QtCore.QSize")
    def test_grid_size(self, mock_qsize):
        mock_qsize.side_effect = [(315, 236.25), (157, 117.75), (157, 117.75)]

        self.assertEqual(get_grid_size(1), mock_qsize.call_args.args)
        self.assertEqual(get_grid_size(5), mock_qsize.call_args.args)
        self.assertEqual(get_grid_size(7, 320, 480), mock_qsize.call_args.args)

    @patch("libs.utils.os.popen")
    def test_posix_file(self, mock_popen):
        file_id = "foobar"
        command = """osascript -e \'get posix path of posix file "{}" -- kthxbai\'""".format("foobar")

        mock_popen.return_value.read.return_value = command

        self.assertEqual(get_posix_file(file_id), command)

if __name__ == "__main__":
    main()
