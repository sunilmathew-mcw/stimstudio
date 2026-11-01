import unittest

from stimulus_forge.services.workflow_agent import (
    command_from_message,
    concept_from_message,
)


class WorkflowChatTests(unittest.TestCase):
    def test_extracts_concept_from_natural_add_command(self) -> None:
        self.assertEqual(
            concept_from_message('Please add concept “Formula 1”'),
            'Formula 1',
        )

    def test_plain_message_is_a_concept(self) -> None:
        self.assertEqual(
            concept_from_message('electric guitar'),
            'electric guitar',
        )

    def test_parses_targeted_related_concept_deletion(self) -> None:
        command = command_from_message(
            'Could you delete related concept Michael Jordan from basketball?'
        )
        self.assertEqual(command.action, 'delete_related')
        self.assertEqual(command.target, 'Michael Jordan')
        self.assertEqual(command.parent, 'basketball')

    def test_parses_clear_all_related_concepts(self) -> None:
        command = command_from_message('delete all the related concepts retrieved')
        self.assertEqual(command.action, 'clear_all_related')

    def test_parses_rename_related_concept(self) -> None:
        command = command_from_message(
            'rename related concept MJ to Michael Jordan in basketball'
        )
        self.assertEqual(command.target, 'MJ')
        self.assertEqual(command.new_name, 'Michael Jordan')
        self.assertEqual(command.parent, 'basketball')

    def test_parses_related_image_request_before_parent_request(self) -> None:
        command = command_from_message(
            'create images for Michael Jordan from basketball'
        )
        self.assertEqual(command.action, 'create_related_images')
        self.assertEqual(command.target, 'Michael Jordan')
        self.assertEqual(command.parent, 'basketball')

    def test_parses_add_related_concept(self) -> None:
        command = command_from_message(
            'add related concept Steph Curry to basketball'
        )
        self.assertIsNotNone(command)
        self.assertEqual(command.action, 'add_related')
        self.assertEqual(command.target, 'Steph Curry')
        self.assertEqual(command.parent, 'basketball')


if __name__ == '__main__':
    unittest.main()
