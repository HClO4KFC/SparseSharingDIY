import pretty_errors

pretty_errors.configure(
    separator_character='*',  # Use '*' as the separator character
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,  # Show line number first
    display_link=True,  # Display file link
    lines_before=5,  # Display 5 lines before error
    lines_after=2,  # Display 2 lines after error
    line_color=pretty_errors.RED + '> ' + pretty_errors.default_config['line_color'],
    code_color='  ' + pretty_errors.default_config['code_color'],
    truncate_code=True,
    display_locals=True
)
