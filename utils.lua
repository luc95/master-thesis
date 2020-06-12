local progress_str = ''
function display_progress(str)
    io.write(('\b \b'):rep(#progress_str))
    io.write(str)
    io.flush()
    progress_str = str
end